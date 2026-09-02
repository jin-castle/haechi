import argparse
import glob
import importlib.util
import os
import subprocess
import sys
from fcntl import ioctl


LSM6DSV16X_WHO_AM_I = 0x0F
LSM6DSV16X_WHO_AM_I_VALUE = 0x70
I2C_SLAVE = 0x0703


class LinuxI2CBus:
    def __init__(self, bus_id):
        self.fd = os.open("/dev/i2c-%d" % bus_id, os.O_RDWR)
        self.current_address = None

    def _select(self, address):
        if self.current_address != address:
            ioctl(self.fd, I2C_SLAVE, address)
            self.current_address = address

    def read_byte_data(self, address, register):
        self._select(address)
        os.write(self.fd, bytes([register & 0xFF]))
        return os.read(self.fd, 1)[0]

    def close(self):
        os.close(self.fd)


def module_available(name):
    return importlib.util.find_spec(name) is not None


def load_smbus():
    for name in ("smbus", "smbus2"):
        if module_available(name):
            return __import__(name)
    class LinuxI2CModule:
        SMBus = LinuxI2CBus

    return LinuxI2CModule


def run_ros2_topic_list():
    try:
        result = run_command(["ros2", "topic", "list"])
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    return set(result.stdout.splitlines())


def run_command(command):
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def read_ros_param(node_name, param_name):
    try:
        result = run_command(["ros2", "param", "get", node_name, param_name])
    except Exception:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if ":" not in output:
        return output
    return output.split(":", 1)[1].strip().strip("'\"")


def parse_int(value):
    return int(value, 0)


def probe_lsm6dsv16x(i2c_bus, i2c_address):
    smbus = load_smbus()
    device_path = "/dev/i2c-%d" % i2c_bus
    if not os.path.exists(device_path):
        return False, "%s is missing" % device_path
    try:
        bus = smbus.SMBus(i2c_bus)
        try:
            who_am_i = bus.read_byte_data(i2c_address, LSM6DSV16X_WHO_AM_I)
        finally:
            bus.close()
    except Exception as exc:
        return False, str(exc)
    if who_am_i != LSM6DSV16X_WHO_AM_I_VALUE:
        return False, "WHO_AM_I got 0x%02X, expected 0x%02X" % (
            who_am_i,
            LSM6DSV16X_WHO_AM_I_VALUE,
        )
    return True, "WHO_AM_I 0x%02X" % who_am_i


def status_line(name, ok, detail):
    label = "OK" if ok else "FAIL"
    return "%s %s: %s" % (label, name, detail)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2c-bus", type=parse_int, default=7)
    parser.add_argument("--i2c-address", type=parse_int, default=0x6A)
    parser.add_argument("--probe-lsm6dsv16x", action="store_true")
    parser.add_argument("--require-real-imu", action="store_true")
    args = parser.parse_args(argv)

    serial_devices = sorted(glob.glob("/dev/ttyUSB*"))
    i2c_devices = sorted(glob.glob("/dev/i2c-*"))
    topics = run_ros2_topic_list()
    smbus_ok = module_available("smbus") or module_available("smbus2")
    python_i2c_detail = "smbus/smbus2 available" if smbus_ok else "linux ioctl fallback available"
    imu_driver = read_ros_param("/imu_node", "driver")
    imu_is_real = imu_driver not in {None, "mock"}

    checks = [
        ("serial", len(serial_devices) >= 2, " ".join(serial_devices) or "no /dev/ttyUSB*"),
        ("i2c", bool(i2c_devices), " ".join(i2c_devices) or "no /dev/i2c-*"),
        ("python_i2c", True, python_i2c_detail),
        ("radar_topic", "/ti_mmwave/radar_scan" in topics, "/ti_mmwave/radar_scan"),
        ("imu_topic", "/imu" in topics, "/imu"),
        ("obstacle_topic", "/firesight/mmwave/front_obstacles" in topics, "/firesight/mmwave/front_obstacles"),
    ]
    if "/imu" in topics or args.require_real_imu:
        checks.append(("imu_driver", imu_driver is not None, imu_driver or "unavailable"))
    if args.require_real_imu:
        checks.append(("real_imu", imu_is_real, imu_driver or "no /imu_node driver parameter"))

    if args.probe_lsm6dsv16x:
        ok, detail = probe_lsm6dsv16x(args.i2c_bus, args.i2c_address)
        checks.append(("lsm6dsv16x", ok, detail))

    for name, ok, detail in checks:
        print(status_line(name, ok, detail))

    hard_fail = any(not ok for name, ok, _ in checks if name in {"serial", "i2c", "python_i2c"})
    if args.probe_lsm6dsv16x:
        hard_fail = hard_fail or any(not ok for name, ok, _ in checks if name == "lsm6dsv16x")
    if args.require_real_imu:
        hard_fail = hard_fail or any(not ok for name, ok, _ in checks if name in {"imu_driver", "real_imu"})
    return 2 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
