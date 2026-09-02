#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

try:
    from .teed_model import TED
except (ImportError, ValueError):
    from teed_model import TED


MEAN_BGR = np.asarray((103.939, 116.779, 123.68), dtype=np.float32)
IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
PROFILE_PROTOCOL = "fire360_video_profile_v1"
FIRE360_OSD_REGIONS = (
    (0.13, 0.72, 0.27, 1.0),
    (0.75, 0.0, 0.90, 0.90),
)


def default_checkpoint_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(script_dir, "..", "..", "models", "teed", "5_model.pth")
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time TEED edge overlay for the original Jetson Nano."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="USB camera index, video/image path, or csi://SENSOR_ID",
    )
    parser.add_argument("--checkpoint", default=default_checkpoint_path())
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--background-scale", type=float, default=0.24)
    parser.add_argument(
        "--contrast-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE strength for low-visibility model input; 0 disables",
    )
    parser.add_argument("--edge-width", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--profiles", help="optional Fire360 video_profiles.json")
    parser.add_argument(
        "--video-key", help="profile input_file key for image/camera input"
    )
    parser.add_argument("--output", help="optional output .png or video path")
    parser.add_argument(
        "--metrics",
        default="jetson_teed_metrics.json",
        help="JSON summary path; use an empty string to disable",
    )
    parser.add_argument("--headless", action="store_true")
    return parser


def validate_args(args):
    if args.width < 64 or args.height < 64:
        raise ValueError("--width and --height must be at least 64")
    if args.capture_width < 64 or args.capture_height < 64:
        raise ValueError("capture dimensions must be at least 64")
    if args.capture_fps < 1:
        raise ValueError("--capture-fps must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if not 0.0 < args.background_scale <= 1.0:
        raise ValueError("--background-scale must be in (0, 1]")
    if args.contrast_clip_limit < 0.0 or args.contrast_clip_limit > 10.0:
        raise ValueError("--contrast-clip-limit must be between 0 and 10")
    if args.edge_width < 1 or args.edge_width > 15:
        raise ValueError("--edge-width must be between 1 and 15")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def should_ignore_fire360_osd(profile_path, video_key):
    if not profile_path:
        return False
    if not os.path.isfile(profile_path):
        raise RuntimeError("missing Fire360 profile: {}".format(profile_path))
    with open(profile_path, "r") as profile_file:
        payload = json.load(profile_file)
    if not isinstance(payload, dict) or payload.get("protocol") != PROFILE_PROTOCOL:
        raise RuntimeError("unsupported Fire360 profile: {}".format(profile_path))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise RuntimeError("Fire360 profile has no profiles list")
    candidate = os.path.basename(video_key).lower()
    for entry in profiles:
        if not isinstance(entry, dict):
            raise RuntimeError("Fire360 profile contains an invalid entry")
        input_file = entry.get("input_file")
        ignores_osd = entry.get("ignore_fire360_osd")
        if not isinstance(input_file, str) or not isinstance(ignores_osd, bool):
            raise RuntimeError("Fire360 profile entry is missing required fields")
        if os.path.basename(input_file).lower() == candidate:
            return ignores_osd
    return False


def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError("missing TEED checkpoint: {}".format(checkpoint_path))
    model = TED().to(device)
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise RuntimeError("unsupported TEED checkpoint: {}".format(checkpoint_path))
    state = {}
    for key, value in loaded.items():
        normalized = key[7:] if key.startswith("module.") else key
        state[normalized] = value
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def csi_pipeline(sensor_id, width, height, fps):
    return (
        "nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM),width={width},height={height},"
        "format=NV12,framerate={fps}/1 ! nvvidconv ! "
        "video/x-raw,width={width},height={height},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    ).format(sensor_id=sensor_id, width=width, height=height, fps=fps)


def open_source(args):
    source = args.source
    if source.lower().startswith("csi://"):
        sensor_text = source.split("://", 1)[1] or "0"
        if not sensor_text.isdigit():
            raise RuntimeError("CSI source must look like csi://0")
        pipeline = csi_pipeline(
            int(sensor_text),
            args.capture_width,
            args.capture_height,
            args.capture_fps,
        )
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        source_kind = "csi_camera"
    elif source.isdigit():
        capture = cv2.VideoCapture(int(source))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
        capture.set(cv2.CAP_PROP_FPS, args.capture_fps)
        source_kind = "usb_camera"
    else:
        capture = cv2.VideoCapture(source)
        source_kind = "video"
    if not capture.isOpened():
        raise RuntimeError("unable to open source: {}".format(source))
    return capture, source_kind


def prepare_tensor(frame, width, height, device, contrast_clip_limit=0.0):
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    model_input = resized
    if contrast_clip_limit > 0.0:
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=contrast_clip_limit,
            tileGridSize=(8, 8),
        )
        enhanced_lightness = clahe.apply(lightness)
        enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
        model_input = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    array = model_input.astype(np.float32)
    array -= MEAN_BGR
    tensor = torch.from_numpy(array.transpose((2, 0, 1))).float().unsqueeze(0)
    pad_width = (8 - width % 8) % 8
    pad_height = (8 - height % 8) % 8
    if pad_width or pad_height:
        tensor = functional.pad(
            tensor,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )
    return resized, tensor.to(device, non_blocking=True)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_fire360_osd(mask):
    height, width = mask.shape
    for left, top, right, bottom in FIRE360_OSD_REGIONS:
        mask[
            round(top * height) : round(bottom * height) + 1,
            round(left * width) : round(right * width) + 1,
        ] = 0
    return mask


def infer(model, frame, args, device, ignore_fire360_osd=False):
    started_total = time.perf_counter()
    resized, tensor = prepare_tensor(
        frame,
        args.width,
        args.height,
        device,
        args.contrast_clip_limit,
    )
    synchronize(device)
    started_model = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor, single_test=True)
        probability = torch.sigmoid(outputs[-1])[0, 0, : args.height, : args.width]
    synchronize(device)
    model_ms = (time.perf_counter() - started_model) * 1000.0
    probability = probability.detach().cpu().numpy()
    mask = (probability >= args.threshold).astype(np.uint8) * 255
    if args.edge_width > 1:
        kernel_size = args.edge_width
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if ignore_fire360_osd:
        mask = clear_fire360_osd(mask)
    background = np.clip(
        resized.astype(np.float32) * args.background_scale,
        0.0,
        255.0,
    ).astype(np.uint8)
    overlay = background
    overlay[mask > 0] = (0, 255, 0)
    total_ms = (time.perf_counter() - started_total) * 1000.0
    return overlay, mask, model_ms, total_ms


def annotate(frame, frame_index, model_ms, total_ms, device):
    text = "TEED {} | model {:.1f} ms | total {:.1f} ms | frame {}".format(
        device, model_ms, total_ms, frame_index
    )
    cv2.putText(
        frame,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def summarize(
    args,
    device,
    source_kind,
    model_times,
    total_times,
    elapsed_seconds,
    video_key,
    ignore_fire360_osd,
):
    frames = len(total_times)
    return {
        "protocol": "teed_jetson_nano_runtime_v1",
        "source": args.source,
        "source_kind": source_kind,
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "resolution": [args.width, args.height],
        "threshold": args.threshold,
        "contrast_clip_limit": args.contrast_clip_limit,
        "profile_path": os.path.abspath(args.profiles) if args.profiles else None,
        "video_key": video_key,
        "ignored_region_count": len(FIRE360_OSD_REGIONS) if ignore_fire360_osd else 0,
        "processed_frames": frames,
        "effective_fps": frames / max(elapsed_seconds, 1e-9),
        "model_latency_ms": {
            "mean": sum(model_times) / max(len(model_times), 1),
            "p50": percentile(model_times, 0.50),
            "p95": percentile(model_times, 0.95),
            "max": max(model_times) if model_times else 0.0,
        },
        "total_latency_ms": {
            "mean": sum(total_times) / max(len(total_times), 1),
            "p50": percentile(total_times, 0.50),
            "p95": percentile(total_times, 0.95),
            "max": max(total_times) if total_times else 0.0,
        },
    }


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("unable to create output video: {}".format(path))
    return writer


def run(args):
    validate_args(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = load_model(args.checkpoint, device)
    print("TEED device: {}".format(device))
    video_key = args.video_key or os.path.basename(args.source)
    ignore_fire360_osd = should_ignore_fire360_osd(args.profiles, video_key)

    image_source = os.path.splitext(args.source)[1].lower() in IMAGE_SUFFIXES
    capture = None
    source_kind = "image"
    if image_source:
        still_frame = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if still_frame is None:
            raise RuntimeError("unable to open image: {}".format(args.source))
    else:
        capture, source_kind = open_source(args)
        success, still_frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError("source produced no frames: {}".format(args.source))

    for _ in range(args.warmup):
        infer(model, still_frame, args, device, ignore_fire360_osd)

    writer = None
    if args.output and not image_source:
        writer = open_writer(
            args.output,
            args.capture_fps,
            (args.width, args.height),
        )
    model_times = []
    total_times = []
    frame_index = 0
    started_at = time.perf_counter()
    current_frame = still_frame
    try:
        while True:
            overlay, _, model_ms, total_ms = infer(
                model,
                current_frame,
                args,
                device,
                ignore_fire360_osd,
            )
            model_times.append(model_ms)
            total_times.append(total_ms)
            annotate(overlay, frame_index, model_ms, total_ms, device)
            if writer is not None:
                writer.write(overlay)
            if not args.headless:
                cv2.imshow("TEED Jetson Nano", overlay)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_index += 1
            if image_source or (
                args.max_frames is not None and frame_index >= args.max_frames
            ):
                if args.output and image_source:
                    if not cv2.imwrite(args.output, overlay):
                        raise RuntimeError(
                            "unable to write output image: {}".format(args.output)
                        )
                break
            success, current_frame = capture.read()
            if not success:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    payload = summarize(
        args,
        device,
        source_kind,
        model_times,
        total_times,
        time.perf_counter() - started_at,
        video_key,
        ignore_fire360_osd,
    )
    if args.metrics:
        with open(args.metrics, "w") as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
            metrics_file.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv=None):
    try:
        run(build_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write("error: {}\n".format(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
