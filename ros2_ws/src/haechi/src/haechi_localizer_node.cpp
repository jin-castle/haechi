#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>
#include <functional>
#include <sstream>
#include <iomanip>
#include <map>
#include <limits>

class HaechiLocalizerNode : public rclcpp::Node {
public:
  HaechiLocalizerNode() : Node("haechi_localizer_node") {
    // --- Callback groups
    cbg_pointcloud_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_pose_imu_radar       = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_pose_imu_thermal     = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_fusion_     = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // --- QoS
    auto pc_qos   = rclcpp::SensorDataQoS();
    auto pose_imu_radar_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    auto pose_imu_thermal_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    // --- SubscriptionOptions (각 콜백 그룹 지정)
    rclcpp::SubscriptionOptions pc_opts;
    pc_opts.callback_group = cbg_pointcloud_;

    rclcpp::SubscriptionOptions pose_imu_radar_opts;
    pose_imu_radar_opts.callback_group = cbg_pose_imu_radar;

    rclcpp::SubscriptionOptions pose_imu_thermal_opts;
    pose_imu_thermal_opts.callback_group = cbg_pose_imu_thermal;

    // ### Subscribers
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/ti_mmwave/radar_scan_pcl", pc_qos,
      std::bind(&HaechiLocalizerNode::onPointCloud, this, std::placeholders::_1),
      pc_opts);

    pose_imu_radar_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/fusion_imu_radar/pose", pose_imu_radar_qos,
      std::bind(&HaechiLocalizerNode::onPoseImuRadar, this, std::placeholders::_1),
      pose_imu_radar_opts);

    pose_imu_thermal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/fusion_imu_thermal/pose", pose_imu_thermal_qos,
      std::bind(&HaechiLocalizerNode::onPoseImuThermal, this, std::placeholders::_1),
      pose_imu_thermal_opts);

    // ### Publishers
    // --- Local pose publisher (최종 로컬 라이제이션 결과)
    local_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/haechi/local_pose", 10);

    // --- Fusion timer (예: 50Hz) : 여기서 센서 융합 수행
    fusion_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),
      std::bind(&HaechiLocalizerNode::fusionTick, this),
      cbg_fusion_);

    RCLCPP_INFO(this->get_logger(),
      "HaechiLocalizerNode started. Subscribing pointcloud & estimated_pose, publishing /haechi/local_pose");
  }

private:
  // ===== 유틸 함수 (기존 코드 그대로) =====
  static const char* datatypeToStr(uint8_t dt) {
    switch (dt) {
      case sensor_msgs::msg::PointField::INT8:   return "int8";
      case sensor_msgs::msg::PointField::UINT8:  return "uint8";
      case sensor_msgs::msg::PointField::INT16:  return "int16";
      case sensor_msgs::msg::PointField::UINT16: return "uint16";
      case sensor_msgs::msg::PointField::INT32:  return "int32";
      case sensor_msgs::msg::PointField::UINT32: return "uint32";
      case sensor_msgs::msg::PointField::FLOAT32:return "float32";
      case sensor_msgs::msg::PointField::FLOAT64:return "float64";
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

  // ===== PointCloud 콜백 =====
  void onPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    const size_t width  = static_cast<size_t>(msg->width);
    const size_t height = static_cast<size_t>(msg->height == 0 ? 1 : msg->height);
    const size_t points = width * height;
    const size_t bytes  = msg->data.size();

    const rclcpp::Time stamp(msg->header.stamp);
    const rclcpp::Time now   = this->get_clock()->now();
    const int64_t      now_ns  = now.nanoseconds();
    const long         now_sec = static_cast<long>(now_ns / 1000000000LL);
    const long         now_nsec= static_cast<long>(now_ns % 1000000000LL);
    const double       latency_s = (now - stamp).seconds();

    // 프레임/포맷
    const bool bigendian = msg->is_bigendian;
    const bool is_dense  = msg->is_dense;
    const uint32_t point_step = msg->point_step;
    const uint32_t row_step   = msg->row_step;
    
    // 필드 요약
    const std::string fields_str = summarizeFields(*msg);

    // 필드 존재 여부
    const bool has_x = hasField(*msg, "x");
    const bool has_y = hasField(*msg, "y");
    const bool has_z = hasField(*msg, "z");
    const bool has_intensity = hasField(*msg, "intensity");
    const bool has_range     = hasField(*msg, "range");
    const bool has_noise     = hasField(*msg, "noise");

    sensor_msgs::PointCloud2ConstIterator<float> it_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*msg, "z");

    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_intensity;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_velocity;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_range;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_snr;
    std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> it_noise;

    if (has_intensity)
      it_intensity = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "intensity");
    if (hasField(*msg, "velocity"))
      it_velocity  = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "velocity");
    else if (hasField(*msg, "doppler"))
      it_velocity  = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "doppler");
    if (has_range)
      it_range     = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "range");
    if (hasField(*msg, "snr"))
      it_snr       = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "snr");
    else if (hasField(*msg, "SNR"))
      it_snr       = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "SNR");
    if (has_noise)
      it_noise     = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "noise");

    // 앞에서부터 N개만 출력
    constexpr size_t N = 5;
    size_t printed = 0;
    std::ostringstream sample_oss;
    sample_oss << std::fixed << std::setprecision(3);

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
        auto s = sample_oss.str();
        if (!s.empty() && s.back() == ',') {
          s.pop_back();
          sample_oss.str("");
          sample_oss.clear();
          sample_oss << s;
        }
        sample_oss << ") ";
        printed++;
      }
    }

    // 이전 클라우드와의 dt (입력 주기 추정)
    double input_dt = std::numeric_limits<double>::quiet_NaN();
    if (has_prev_cloud_stamp_)
    {
      input_dt = (stamp - prev_cloud_stamp_).seconds();
    }
    prev_cloud_stamp_ = stamp;
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

    if (printed > 0) {
      RCLCPP_DEBUG(get_logger(), "  samples(first %zu): %s",
                   printed, sample_oss.str().c_str());
    }

    // 공유 버퍼 갱신
    {
      std::lock_guard<std::mutex> lock(mutex_);
      last_cloud_ = msg;
      cloud_received_ = true;
    }
  }

  // ===== Pose w/ IMU, Radar 콜백 =====
  void onPoseImuRadar(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  const auto & t = msg->pose.position;
  const auto & q = msg->pose.orientation;

  const rclcpp::Time stamp(msg->header.stamp);
  const double latency_s = (this->now() - stamp).seconds();

  RCLCPP_INFO(get_logger(),
    "[ImuRararPose] header.frame_id=%s time=%d.%09u latency=%.3f s "
    "pos=(%.3f, %.3f, %.3f) quat=(%.3f, %.3f, %.3f, %.3f)",
    msg->header.frame_id.c_str(),
    msg->header.stamp.sec, msg->header.stamp.nanosec, latency_s,
    t.x, t.y, t.z, q.x, q.y, q.z, q.w);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      last_pose_ = msg;
      pose_received_ = true;
    }
  }

  // ===== Pose w/ IMU, Thermal 콜백 =====
  void onPoseImuThermal(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  const auto & t = msg->pose.position;
  const auto & q = msg->pose.orientation;

  const rclcpp::Time stamp(msg->header.stamp);
  const double latency_s = (this->now() - stamp).seconds();

  RCLCPP_INFO(get_logger(),
    "[ImuThermal] header.frame_id=%s time=%d.%09u latency=%.3f s "
    "pos=(%.3f, %.3f, %.3f) quat=(%.3f, %.3f, %.3f, %.3f)",
    msg->header.frame_id.c_str(),
    msg->header.stamp.sec, msg->header.stamp.nanosec, latency_s,
    t.x, t.y, t.z, q.x, q.y, q.z, q.w);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      last_pose_ = msg;
      pose_received_ = true;
    }
  }

  // ===== Fusion 타이머: 로컬 포즈 계산 & 퍼블리시 =====
  void fusionTick() {
  std::shared_ptr<sensor_msgs::msg::PointCloud2> cloud;
  std::shared_ptr<geometry_msgs::msg::PoseStamped> pose;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!(cloud_received_ && pose_received_)) {
      // 아직 둘 다 안 들어옴
      return;
    }
    cloud = last_cloud_;
    pose  = last_pose_;
  }

  // TODO: 여기서 cloud + pose를 이용해서 실제 로컬라이제이션 수행
  // 지금은 imu/pose를 그대로 odom 프레임으로 패스스루

  geometry_msgs::msg::PoseStamped local_pose;
  local_pose.header = pose->header;
  local_pose.header.frame_id = "odom";  // 로컬 좌표계 이름 (필요하면 파라미터화)
  local_pose.pose = pose->pose;         // position & orientation 통째로 복사

  local_pose_pub_->publish(local_pose);

  RCLCPP_DEBUG(get_logger(),
    "[FusionTick] publish local_pose t=%d.%09u",
    local_pose.header.stamp.sec,
    local_pose.header.stamp.nanosec);
}

private:
  // Subscriptions
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_imu_radar_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_imu_thermal_sub_;

  // Publisher
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr local_pose_pub_;

  // Callback groups
  rclcpp::CallbackGroup::SharedPtr cbg_pointcloud_;
  rclcpp::CallbackGroup::SharedPtr cbg_pose_imu_radar;
  rclcpp::CallbackGroup::SharedPtr cbg_pose_imu_thermal;
  rclcpp::CallbackGroup::SharedPtr cbg_fusion_;

  // Fusion timer
  rclcpp::TimerBase::SharedPtr fusion_timer_;

  // Shared buffers for fusion
  std::mutex mutex_;
  std::shared_ptr<sensor_msgs::msg::PointCloud2> last_cloud_;
  std::shared_ptr<geometry_msgs::msg::PoseStamped> last_pose_;
  bool cloud_received_{false};
  bool pose_received_{false};

  rclcpp::Time prev_cloud_stamp_;
  bool has_prev_cloud_stamp_{false};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HaechiLocalizerNode>();

  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(), /*number_of_threads=*/2);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
