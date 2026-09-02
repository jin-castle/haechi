#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
// 맵 타입은 프로젝트에 맞게: 예시로 PointCloud2 사용
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <mutex>
#include <memory>
#include <chrono>

class HaechiMapMatcherNode : public rclcpp::Node {
public:
  HaechiMapMatcherNode() : Node("haechi_map_matcher_node") {
    // 콜백 그룹
    cbg_local_pose_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_map_        = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    cbg_match_      = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions local_pose_opts;
    local_pose_opts.callback_group = cbg_local_pose_;

    rclcpp::SubscriptionOptions map_opts;
    map_opts.callback_group = cbg_map_;

    // 로컬 포즈 구독
    local_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/haechi/local_pose", 10,
      std::bind(&HaechiMapMatcherNode::onLocalPose, this, std::placeholders::_1),
      local_pose_opts);

    // 외부 맵 구독 (토픽/타입은 실제 시스템에 맞게 수정)
    map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/haechi/map_cloud", 1,
      std::bind(&HaechiMapMatcherNode::onMapCloud, this, std::placeholders::_1),
      map_opts);

    // 글로벌 포즈 퍼블리셔
    global_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/haechi/global_pose", 10);

    // 매칭 타이머 (예: 2Hz)
    match_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&HaechiMapMatcherNode::matchTick, this),
      cbg_match_);

    RCLCPP_INFO(this->get_logger(),
      "HaechiMapMatcherNode started. Subscribing /haechi/local_pose and /haechi/map_cloud, publishing /haechi/global_pose");
  }

private:
  void onLocalPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_local_pose_ = msg;
    local_pose_received_ = true;
  }

  void onMapCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_map_cloud_ = msg;
    map_received_   = true;
  }

  // 매칭 타이머: 로컬 포즈 + 맵으로 글로벌 포즈 계산
  void matchTick() {
    std::shared_ptr<geometry_msgs::msg::PoseStamped> local_pose;
    std::shared_ptr<sensor_msgs::msg::PointCloud2> map_cloud;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!(local_pose_received_ && map_received_)) {
        // 데이터 아직 준비 안 됨
        return;
      }
      local_pose = last_local_pose_;
      map_cloud  = last_map_cloud_;
    }

    // TODO: 여기서 map matching / scan-to-map / pose graph optimization 등 수행
    // 지금은 예시로 "로컬 포즈를 그대로 글로벌 포즈로 내보내는" 패스스루 구현

    auto global_pose = *local_pose;
    global_pose.header.frame_id = "map";  // 글로벌 좌표계 이름

    global_pose_pub_->publish(global_pose);

    RCLCPP_DEBUG(get_logger(),
      "[matchTick] publish global_pose t=%d.%09u",
      global_pose.header.stamp.sec,
      global_pose.header.stamp.nanosec);
  }

private:
  // Subscriptions
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr local_pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr   map_sub_;

  // Publisher
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr global_pose_pub_;

  // Callback groups
  rclcpp::CallbackGroup::SharedPtr cbg_local_pose_;
  rclcpp::CallbackGroup::SharedPtr cbg_map_;
  rclcpp::CallbackGroup::SharedPtr cbg_match_;

  // Timer
  rclcpp::TimerBase::SharedPtr match_timer_;

  // 공유 버퍼
  std::mutex mutex_;
  std::shared_ptr<geometry_msgs::msg::PoseStamped>  last_local_pose_;
  std::shared_ptr<sensor_msgs::msg::PointCloud2>    last_map_cloud_;
  bool local_pose_received_{false};
  bool map_received_{false};
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HaechiMapMatcherNode>();

  // 맵 매칭은 연산이 조금 무거울 수 있으니, 필요하면 스레드 수 2~3으로 조정 가능
  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(), /*number_of_threads=*/2);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}

