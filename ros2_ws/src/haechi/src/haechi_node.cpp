#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>
#include <functional>
#include <sstream>
#include <iomanip>
#include <map>
#include <limits>

class HaechiNode : public rclcpp::Node {
public:
  HaechiNode() : Node("haechi_node") {
    // --- Callback groups: 서로 다른 콜백이 병렬 실행될 수 있도록 Reentrant 사용
    cbg_pointcloud_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_pose_       = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_fusion_     = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // --- QoS
    auto pc_qos = rclcpp::SensorDataQoS(); // 고주파 센서 데이터
    auto pose_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    // --- SubscriptionOptions로 각 콜백 그룹 지정
    rclcpp::SubscriptionOptions pc_opts;
    pc_opts.callback_group = cbg_pointcloud_;

    rclcpp::SubscriptionOptions pose_opts;
    pose_opts.callback_group = cbg_pose_;

    // --- Subscribers
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/ti_mmwave/radar_scan_pcl", pc_qos,
      std::bind(&HaechiNode::onPointCloud, this, std::placeholders::_1),
      pc_opts);

    pose_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
      "estimated_pose", pose_qos,
      std::bind(&HaechiNode::onPose, this, std::placeholders::_1),
      pose_opts);

    // --- Fusion timer (예: 50Hz). 향후 실제 센서 퓨전 로직을 여기로.
  // 교체 후 (Humble 호환: 콜백그룹을 직접 인자로 전달)
  fusion_timer_ = this->create_wall_timer(
  std::chrono::milliseconds(20),
  std::bind(&HaechiNode::fusionTick, this),
  cbg_fusion_);   // ✅ callback group

    RCLCPP_INFO(this->get_logger(), "Haechi node started. Subscribing to 'pointcloud' and 'estimated_pose'.");
  }

private:
  // ---- 유틸: 필드 타입 문자열화
  static const char* datatypeToStr(uint8_t dt) {
    switch (dt) {
      case sensor_msgs::msg::PointField::INT8: return "int8";
      case sensor_msgs::msg::PointField::UINT8: return "uint8";
      case sensor_msgs::msg::PointField::INT16: return "int16";
      case sensor_msgs::msg::PointField::UINT16: return "uint16";
      case sensor_msgs::msg::PointField::INT32: return "int32";
      case sensor_msgs::msg::PointField::UINT32: return "uint32";
      case sensor_msgs::msg::PointField::FLOAT32: return "float32";
      case sensor_msgs::msg::PointField::FLOAT64: return "float64";
      default: return "unknown";
    }
  }

  static bool hasField(const sensor_msgs::msg::PointCloud2 &msg, const std::string &name) {
    for (const auto &f : msg.fields) if (f.name == name) return true;
    return false;
  }

  static std::string summarizeFields(const sensor_msgs::msg::PointCloud2 &msg) {
    std::ostringstream oss;
    for (size_t i = 0; i < msg.fields.size(); ++i) {
      const auto &f = msg.fields[i];
      oss << f.name << "(" << datatypeToStr(f.datatype) << "@"
          << f.offset << ",count=" << f.count << ")";
      if (i + 1 < msg.fields.size()) oss << ", ";
    }
    return oss.str();
  }

  // ---- PointCloud 콜백: 확장 로그
  void onPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 기본 정보
    const size_t width  = static_cast<size_t>(msg->width);
    const size_t height = static_cast<size_t>(msg->height == 0 ? 1 : msg->height);
    const size_t points = width * height;
    const size_t bytes  = msg->data.size();

    // 시간/지연
    const rclcpp::Time stamp(msg->header.stamp);
    const rclcpp::Time now = this->get_clock()->now();
    const int64_t now_ns   = now.nanoseconds();
    const long    now_sec  = static_cast<long>(now_ns / 1000000000LL);
    const long    now_nsec = static_cast<long>(now_ns % 1000000000LL);
    const double latency_s   = (now - stamp).seconds();

    // now를 ROS clock에서 얻기 (stamp와 같은 clock)
    


    // 프레임/포맷
    const bool bigendian = msg->is_bigendian;
    const bool is_dense  = msg->is_dense;
    const uint32_t point_step = msg->point_step; // bytes per point
    const uint32_t row_step   = msg->row_step;   // bytes per row

    // 필드 요약
    const std::string fields_str = summarizeFields(*msg);

    // 첫 N개 샘플 포인트 요약 (가능한 필드만)
    constexpr size_t N = 5;
    size_t printed = 0;

    std::ostringstream sample_oss;
    sample_oss << std::fixed << std::setprecision(3);

    // 존재 여부 확인
    const bool has_x = hasField(*msg, "x");
    const bool has_y = hasField(*msg, "y");
    const bool has_z = hasField(*msg, "z");
    const bool has_intensity = hasField(*msg, "intensity");
    // 레이더일 수 있으니 이름 후보 넓게
    // const bool has_velocity  = hasField(*msg, "velocity") || hasField(*msg, "doppler");
    const bool has_range     = hasField(*msg, "range");
    // const bool has_snr       = hasField(*msg, "snr") || hasField(*msg, "SNR");
    const bool has_noise     = hasField(*msg, "noise");

    // NOTE: 모든 포인트를 도는 건 비용이 크니, 앞에서 최대 N개만
    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");

    // 선택적 필드: 없으면 더미 이터레이터는 만들지 않는다
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_intensity;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_velocity;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_range;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_snr;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_noise;

    if (has_intensity) it_intensity = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "intensity");
    if (hasField(*msg, "velocity")) it_velocity = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "velocity");
    else if (hasField(*msg, "doppler")) it_velocity = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "doppler");
    if (has_range) it_range = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "range");
    if (hasField(*msg, "snr")) it_snr = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "snr");
    else if (hasField(*msg, "SNR")) it_snr = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "SNR");
    if (has_noise) it_noise = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "noise");

    // 앞에서부터 N개만 출력
    if (points > 0) {
      size_t cap = std::min(points, N);
      for (size_t i = 0; i < cap; ++i, ++it_x, ++it_y, ++it_z) {
        sample_oss << "p" << i << "=(";
        if (has_x) sample_oss << "x=" << *it_x << ",";
        if (has_y) sample_oss << "y=" << *it_y << ",";
        if (has_z) sample_oss << "z=" << *it_z << ",";
        if (it_intensity) { sample_oss << "I=" << **it_intensity << ","; ++(*it_intensity); }
        if (it_velocity)  { sample_oss << "V=" << **it_velocity  << ","; ++(*it_velocity);  }
        if (it_range)     { sample_oss << "R=" << **it_range     << ","; ++(*it_range);     }
        if (it_snr)       { sample_oss << "SNR=" << **it_snr     << ","; ++(*it_snr);       }
        if (it_noise)     { sample_oss << "N=" << **it_noise     << ","; ++(*it_noise);     }
        // 끝 콤마 정리
        auto s = sample_oss.str();
        if (!s.empty() && s.back() == ',') { s.pop_back(); sample_oss.str(""); sample_oss.clear(); sample_oss << s; }
        sample_oss << ") ";
        printed++;
      }
    }

    // 이전 클라우드와의 dt (입력 주기 추정)
    double input_dt = std::numeric_limits<double>::quiet_NaN();

    if (has_prev_cloud_stamp_) {                // 이전 값이 있을 때만 비교
      input_dt = (stamp - prev_cloud_stamp_).seconds();
    }
    prev_cloud_stamp_ = stamp;                  // 같은 소스로 저장 (ROS_TIME)
    has_prev_cloud_stamp_ = true;

    RCLCPP_INFO(get_logger(),
    "[PointCloud2]\n"
    "  frame_id=%s  stamp=%d.%09u  latency=%.3f s  now=%ld.%09ld\n"
    "  dims: width=%u height=%u  points=%zu  bytes=%zu\n"
    "  step: point_step=%u row_step=%u  endian=%s  is_dense=%s\n"
    "  fields(%zu): %s\n"
    "  input_dt(est)=%.4f s",
    msg->header.frame_id.c_str(),
    msg->header.stamp.sec, msg->header.stamp.nanosec, latency_s,
    now_sec, now_nsec,
    msg->width, msg->height, points, bytes,
    point_step, row_step, (bigendian ? "big" : "little"), (is_dense ? "true" : "false"),
    msg->fields.size(), fields_str.c_str(),
    input_dt);

    // ------- 샘플 포인트 (DEBUG) : 터미널이 너무 시끄러우면 레벨만 INFO -> DEBUG 유지
    if (printed > 0) {
      RCLCPP_DEBUG(get_logger(), "  samples(first %zu): %s", printed, sample_oss.str().c_str());
    }

    // 공유 버퍼 갱신
    {
      std::lock_guard<std::mutex> lock(mutex_);
      last_cloud_ = msg;
      cloud_received_ = true;
    }
  }


  // Pose 콜백: 기존 그대로 (이미 풍부함)
  void onPose(const geometry_msgs::msg::TransformStamped::SharedPtr tf) {
    const auto & t = tf->transform.translation;
    const auto & q = tf->transform.rotation;

    const rclcpp::Time stamp(tf->header.stamp);
    const double latency_s = (this->now() - stamp).seconds();

    RCLCPP_INFO(get_logger(),
      "[EstimatedPose] header.frame_id=%s child_frame_id=%s time=%d.%09u latency=%.3f s "
      "pos=(%.3f, %.3f, %.3f) quat=(%.3f, %.3f, %.3f, %.3f)",
      tf->header.frame_id.c_str(), tf->child_frame_id.c_str(),
      tf->header.stamp.sec, tf->header.stamp.nanosec, latency_s,
      t.x, t.y, t.z, q.x, q.y, q.z, q.w);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      last_pose_ = tf;
      pose_received_ = true;
    }
  }

  // 퓨전 워커: 현재는 “둘 다 최신 데이터가 있으면 타임스탬프 표시”만 함
  // 실제 퓨전(예: EKF/UKF/Factor Graph)은 여기서 last_*를 읽어 처리하면 됨.
  void fusionTick() {
    std::shared_ptr<sensor_msgs::msg::PointCloud2> cloud;
    std::shared_ptr<geometry_msgs::msg::TransformStamped> pose;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!(cloud_received_ && pose_received_)) {
        return; // 아직 둘 다 안 들어옴
      }
      cloud = last_cloud_;
      pose = last_pose_;
    }

    // 예시 출력: “둘 다 최신”일 때만 간단 로그
    RCLCPP_DEBUG(get_logger(),
      "[FusionTick] cloud t=%d.%09u, pose t=%d.%09u",
      cloud->header.stamp.sec, cloud->header.stamp.nanosec,
      pose->header.stamp.sec,  pose->header.stamp.nanosec);

    // TODO: 여기서 실제 센서 퓨전 로직 수행 (예: voxel downsample + pose 보정 등)
  }

private:
  // Subscriptions
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr pose_sub_;

  // Callback groups
  rclcpp::CallbackGroup::SharedPtr cbg_pointcloud_;
  rclcpp::CallbackGroup::SharedPtr cbg_pose_;
  rclcpp::CallbackGroup::SharedPtr cbg_fusion_;

  // Fusion timer
  rclcpp::TimerBase::SharedPtr fusion_timer_;

  // Shared buffers for fusion
  std::mutex mutex_;
  std::shared_ptr<sensor_msgs::msg::PointCloud2> last_cloud_;
  std::shared_ptr<geometry_msgs::msg::TransformStamped> last_pose_;
  bool cloud_received_{false};
  bool pose_received_{false};

  // Timer
  // rclcpp::Time prev_cloud_stamp_{0, 0};  // 0초 초기값
  rclcpp::Time prev_cloud_stamp_;     // 초기값 신경 X (안 쓸 거라)
  bool has_prev_cloud_stamp_{false};  // 첫 프레임 건너뛰기 플래그
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HaechiNode>();

  // 멀티스레드 실행자: 콜백이 서로 다른 스레드에서 병렬 실행됨
  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(), /*number_of_threads=*/2);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
