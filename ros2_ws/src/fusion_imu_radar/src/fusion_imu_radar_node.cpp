#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include <ti_mmwave_rospkg_msgs/msg/radar_scan.hpp>

#include <mutex>
#include <optional>

class FusionImuRadarNode : public rclcpp::Node
{
public:
    FusionImuRadarNode() : Node("fusion_imu_radar_node")
    {
        //=========== ROS TOPIC ===========//
        // IMU 구독자: /imu
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu",
            rclcpp::SensorDataQoS(),   // IMU라서 SensorData QoS 사용
            std::bind(&FusionImuRadarNode::imuCallback, this, std::placeholders::_1));

        radar_sub_ = this->create_subscription<ti_mmwave_rospkg_msgs::msg::RadarScan>(
            "/ti_mmwave/radar_scan",
            rclcpp::SensorDataQoS(),
            std::bind(&FusionImuRadarNode::radarCallback, this, std::placeholders::_1));

        // radar_pcl_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        //     "/ti_mmwave/radar_scan_pcl",
        //     rclcpp::SensorDataQoS(),
        //     std::bind(&FusionImuRadarNode::radarPclCallback, this, std::placeholders::_1));

        // Pose 발행자: /fusion_imu_radar/pose
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/fusion_imu_radar/pose",
            10);

        RCLCPP_INFO(this->get_logger(), "ROS Topic initialized");


        // --- Timer (fusion + publish 주기)
        // 예: 50Hz(20ms). 필요에 따라 10~200Hz로 조절.
        fusion_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&FusionImuRadarNode::fusionTick, this));

        //=========== EKF init ===========//
        ekf_x_.setZero();

        ekf_P_.setIdentity();
        ekf_P_ *= 1.0f;

        ekf_Q_.setZero();
        ekf_Q_.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * 0.001f;    // position noise
        ekf_Q_.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * 0.05f;     // velocity noise

        ekf_R_radar_.setIdentity();
        ekf_R_radar_ *= 0.05f;  // radar Δ 노이즈

        RCLCPP_INFO(this->get_logger(), "EKF Initialized");
    }

    ~FusionImuRadarNode() {}

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void radarCallback(const ti_mmwave_rospkg_msgs::msg::RadarScan::SharedPtr msg);
    void radarPclCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void fusionTick();

    void ekfPredict(float dt, const Eigen::Vector3f& acc, Eigen::Matrix<float,6,1>& x, Eigen::Matrix<float,6,6>& P, const Eigen::Matrix<float,6,6>& Q);
    // void ekfRadarUpdate(const Eigen::Vector3f &z, Eigen::Matrix<float, 6, 1> &x, Eigen::Matrix<float, 6, 6> &P, const Eigen::Matrix3f &R);
    void ekfRadarUpdate(const Eigen::Vector3f &v_meas, Eigen::Matrix<float, 6, 1> &x, Eigen::Matrix<float, 6, 6> &P, const Eigen::Matrix3f &R);

private:
    //=========== ROS TOPIC ===========//
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<ti_mmwave_rospkg_msgs::msg::RadarScan>::SharedPtr radar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_pcl_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

    //=========== Variables ===========//
    // imu 변수
    bool have_imu_{false};
    rclcpp::Time last_imu_time_{0, 0, RCL_ROS_TIME};
    sensor_msgs::msg::Imu last_imu_;

    // radar 변수
    bool have_radar_{false};
    // bool first_radar_ = true;
    rclcpp::Time last_radar_time_{0, 0, RCL_ROS_TIME};
    // Eigen::Vector3f last_radar_pos_;
    Eigen::Vector3f last_radar_vel_;
    
    struct RadarPoint
    {
        Eigen::Vector3f pos;
        float doppler;
    };
    std::vector<RadarPoint> radar_buffer_;

    // fusion 변수
    std::mutex data_mtx_;
    rclcpp::TimerBase::SharedPtr fusion_timer_;
    rclcpp::Time last_predict_time_{0, 0, RCL_ROS_TIME};

    //=========== EKF states ===========//
    bool ekf_initialized_ = false;

    // x = [px py pz vx vy vz]
    Eigen::Matrix<float, 6, 1> ekf_x_;
    Eigen::Matrix<float, 6, 6> ekf_P_;
    Eigen::Matrix<float, 6, 6> ekf_Q_;
    Eigen::Matrix3f ekf_R_radar_;
};



int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FusionImuRadarNode>();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

void FusionImuRadarNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    // ─────────────────────────────────────────────
    // 0) 동일 timestamp 메시지 중복 처리 방지 (선택)
    // ─────────────────────────────────────────────
    // static rclcpp::Time last_stamp(0, 0, RCL_ROS_TIME);
    // if (msg->header.stamp == last_stamp)
    //   return;
    // last_stamp = msg->header.stamp;
    rclcpp::Time current_time_ = msg->header.stamp;

    // ─────────────────────────────────────────────
    // 1) 최신 IMU 저장 (fusionTick이 읽을 데이터)
    // ─────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(data_mtx_);
        have_imu_ = true;
        last_imu_ = *msg;
        last_imu_time_ = rclcpp::Time(current_time_);
    }
}

void FusionImuRadarNode::radarCallback(const ti_mmwave_rospkg_msgs::msg::RadarScan::SharedPtr msg)
{
    rclcpp::Time current_time_ = msg->header.stamp;

    std::lock_guard<std::mutex> lk(data_mtx_);

    // 새로운 frame 시작
    if (msg->point_id == 0)
    {
        radar_buffer_.clear();
    }

    if (radar_buffer_.empty())
    {
        have_radar_ = true;
        last_radar_time_ = rclcpp::Time(current_time_);
    }

    RadarPoint p;
    p.pos = Eigen::Vector3f(msg->x, msg->y, msg->z);
    p.doppler = msg->velocity;

    radar_buffer_.push_back(p);

    // TODO: 반드시 실제 값으로 교체
    // constexpr float doppler_resolution = 0.043f;
    // float doppler = msg->doppler_bin * doppler_resolution;
    // if (std::abs(doppler) < 0.2f)
    //     return;
    // Eigen::Vector3f p(msg->x, msg->y, msg->z);
    // float r = p.norm();
    // if (r < 0.5f || r > 10.0f)
    //     return;
    // Eigen::Vector3f r_hat = p.normalized();
    // Eigen::Vector3f v_ego_radar = -doppler * r_hat;
    // {
    //     std::lock_guard<std::mutex> lk(data_mtx_);
    //     last_radar_vel_ = v_ego_radar;
    //     last_radar_time_ = msg->header.stamp;
    //     have_radar_ = true;
    // }
}

/*
void FusionImuRadarNode::radarPclCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // ─────────────────────────────────────────────
    // 0) 동일 timestamp 메시지 중복 처리 방지 (선택)
    // ─────────────────────────────────────────────
    // static rclcpp::Time last_stamp(0, 0, RCL_ROS_TIME);
    // if (msg->header.stamp == last_stamp)
    //     return;
    // last_stamp = msg->header.stamp;
    rclcpp::Time current_time_ = msg->header.stamp;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);
    if (cloud.points.empty())
        return;

    // ─────────────────────────────────────────────
    // 1) Point-level filtering
    // ─────────────────────────────────────────────
    std::vector<float> xs, ys, zs;
    xs.reserve(cloud.points.size());
    for (const auto &pt : cloud.points)
    {
        // 거리 필터
        float r = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (r < 0.5f || r > 10.0f)
            continue;

        // z 필터 (바닥/천장/노이즈 제거)
        if (pt.z < -0.2f || pt.z > 0.3f)
            continue;

        xs.push_back(pt.x);
        ys.push_back(pt.y);
        zs.push_back(pt.z);
    }

    if (xs.size() < 5)
    {
        // std::cout << "관측 신뢰 불가" << std::endl;  → EKF update skip
        return;
    }

    // ─────────────────────────────────────────────
    // 2) 최신 Radar 저장 (fusionTick이 읽을 데이터)
    // ─────────────────────────────────────────────
    auto median = [](std::vector<float> &v) -> float
    {
        size_t n = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + n, v.end());
        return v[n];
    };

    // 포착한 x,y,z의 평균값
    float current_x = median(xs);
    float current_y = median(ys);
    float current_z = median(zs);
    Eigen::Vector3f current_pos(current_x, current_y, current_z);

    // 위치 변화량으로 '속도' 추출하기
    if (first_radar_)
    {
        have_radar_ = true; // 이제 fusionTick에서 이 값을 '속도'로 사용함

        last_radar_pos_ = current_pos;
        last_radar_time_ = current_time_;

        first_radar_ = false;
        return;         // 첫 데이터는 비교 대상이 없으므로 저장만 하고 리턴
    }

    double r_dt = (current_time_ - last_radar_time_).seconds();
    if (r_dt > 0.001)     // 시간 간격이 유효할 때만
    {
        std::lock_guard<std::mutex> lk(data_mtx_);

        // 따라서 (이전 거리 - 현재 거리) / dt 가 나의 전진 속도가 됩니다.
        float radar_dx_ = (last_radar_pos_.x() - current_x) / r_dt; // vx_radar
        float radar_dy_ = (last_radar_pos_.y() - current_y) / r_dt; // vy_radar
        float radar_dz_ = (last_radar_pos_.z() - current_z) / r_dt; // vz_radar
        last_radar_vel_ = Eigen::Vector3f(radar_dx_, radar_dy_, radar_dz_);

        last_radar_pos_ = current_pos;
        last_radar_time_ = current_time_;
    }
}
*/

// -------------------------
// Timer: fusion + publish
// -------------------------
void FusionImuRadarNode::fusionTick()
{
    // imu
    sensor_msgs::msg::Imu imu;
    rclcpp::Time imu_t;
    bool process_imu = false;

    // radar
    // float radar_dx, radar_dy, radar_dz;
    std::vector<RadarPoint> radar_frame;
    rclcpp::Time radar_t;
    bool process_radar = false;

    {
        std::lock_guard<std::mutex> lk(data_mtx_);

        if (have_imu_)
        {
            imu = last_imu_;
            imu_t = last_imu_time_;

            process_imu = true;
            have_imu_ = false;      // 소비 완료 (중복 업데이트 방지)
        }

        if (have_radar_)
        {
            radar_frame = radar_buffer_;     // 복사
            radar_buffer_.clear();           // 소비

            radar_t = last_radar_time_;
            process_radar = true;
            have_radar_ = false;
            // std::cout << "radar buffer clear" << std::endl;
        }
    }

    if (ekf_initialized_ == false)
    {
        if (process_imu)
        {
            last_predict_time_ = imu_t;

            ekf_x_.setZero();
            ekf_initialized_ = true;
        }
        return;
    }

    // ===== 1. EKF Predict (IMU) =====
    if (process_imu)
    {
        double dt = (imu_t - last_predict_time_).seconds();

        if (dt > 0.0 && dt < 0.2)
        {
            Eigen::Quaternionf q(imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z);               // 1. IMU 쿼터니언 생성 - imu.orientation을 통해 제공받은 imu의 월드 좌표계에 대한 오리엔테이션 값을 사용
            Eigen::Vector3f imu_acc_body(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);  // 2. 센서 좌표계 데이터(가속도)
            Eigen::Vector3f imu_acc_world = q * imu_acc_body;                                                               // 3. 세계 좌표계로 회전 변환
            imu_acc_world.z() -= 9.81f;                                                                                     // 4. 세계 좌표계 기준 Z축 중력 제거 (9.81) --> acc_world는 월드 좌표계에서에 대한 imu의 선형가속도를 저장

            ekfPredict((float)dt, imu_acc_world, ekf_x_, ekf_P_, ekf_Q_);

            RCLCPP_INFO(this->get_logger(), "Predict!");
        }
        last_predict_time_ = imu_t;
    }

    // ===== 2. EKF Update (Radar) =====
    // 1) ego velocity from radar buffer
    Eigen::Vector3f radar_ego_vel(0,0,0);
    int valid_cnt = 0;

    for (const auto& p : radar_frame)
    {
        float range = p.pos.norm();
        
        // std::cout << "range = " << range << std::endl;

        if (range < 0.2f || range > 5.0f) continue;   // 거리 필터
        if (std::abs(p.pos.z()) > 0.5f) continue;     // 지면/노이즈 제거

        Eigen::Vector3f r_hat = p.pos.normalized();

        // Doppler는 LOS 기준 "상대속도"
        Eigen::Vector3f v = -p.doppler * r_hat;

        if (v.norm() < 5.0f)
        {
            radar_ego_vel += v;
            valid_cnt++;
        }
    }

    if (valid_cnt > 3)
    {
        radar_ego_vel /= valid_cnt;
    }
    else
    {
        process_radar = false;   // 신뢰 불가 → EKF update 안 함
    }
    
    // 2) EKF Update
    // if (process_radar)
    if (valid_cnt > 3 /*10 && radar_ego_vel.norm() < 3.0f && (radar_ego_vel - ekf_x_.segment<3>(3)).norm() < 2.0f*/)
    {
        Eigen::Quaternionf q(last_imu_.orientation.w, last_imu_.orientation.x, last_imu_.orientation.y, last_imu_.orientation.z);

        Eigen::Vector3f radar_vel_world = q * radar_ego_vel;

        ekfRadarUpdate(radar_vel_world, ekf_x_, ekf_P_, ekf_R_radar_);

        RCLCPP_INFO(this->get_logger(), "Update!");
    }
    // else
    // {
    //     std::cout << "valid_cnt: " << valid_cnt << "radar_ego_vel.norm(): " << radar_ego_vel.norm() << "(radar_ego_vel - ekf_x_.segment<3>(3)).norm(): " << (radar_ego_vel - ekf_x_.segment<3>(3)).norm() << std::endl;
    // }

    // ===== 3. Topic Publish =====
    if (process_imu || process_radar)
    {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = this->now();
        pose_msg.header.frame_id = "map"; // 가급적 고정 좌표계 이름 사용

        pose_msg.pose.position.x = ekf_x_(0);
        pose_msg.pose.position.y = ekf_x_(1);
        pose_msg.pose.position.z = ekf_x_(2);

        // IMU의 쿼터니언을 그대로 복사 (Orientation Fusion은 생략된 상태이므로)
        pose_msg.pose.orientation = imu.orientation;
        pose_pub_->publish(pose_msg);

        // fusion 결과 터미널 출력 추가 (0.1초마다 혹은 데이터가 올 때마다) ---
        RCLCPP_INFO(
            this->get_logger(),
            "[FUSION RESULT] X: %.3f, Y: %.3f, Z: %.3f | Vel_X: %.3f, Vel_Y: %.3f, Vel_Z: %.3f",
            ekf_x_(0), ekf_x_(1), ekf_x_(2), ekf_x_(3), ekf_x_(4), ekf_x_(5)
        );
    }
}

void FusionImuRadarNode::ekfPredict(float dt, const Eigen::Vector3f& acc, Eigen::Matrix<float,6,1>& x, Eigen::Matrix<float,6,6>& P, const Eigen::Matrix<float,6,6>& Q)
{
    Eigen::Matrix<float,6,6> F = Eigen::Matrix<float,6,6>::Identity();
    F(0,3) = dt;
    F(1,4) = dt;
    F(2,5) = dt;

    Eigen::Matrix<float,6,3> B = Eigen::Matrix<float,6,3>::Zero();
    B(0,0) = 0.5f * dt * dt;
    B(1,1) = 0.5f * dt * dt;
    B(2,2) = 0.5f * dt * dt;
    B(3,0) = dt;
    B(4,1) = dt;
    B(5,2) = dt;

    x = F * x + B * acc;
    P = F * P * F.transpose() + Q;
}

// void FusionImuRadarNode::ekfRadarUpdate(const Eigen::Vector3f &z, Eigen::Matrix<float, 6, 1> &x, Eigen::Matrix<float, 6, 6> &P, const Eigen::Matrix3f &R)
// {
//   Eigen::Matrix<float, 3, 6> H = Eigen::Matrix<float, 3, 6>::Zero();
//   H(0, 0) = 1.0f;  // px
//   H(1, 1) = 1.0f;  // py
//   H(2, 2) = 1.0f;  // pz
//   Eigen::Vector3f z_pred = H * x;
//   Eigen::Vector3f y = z - z_pred;
//   Eigen::Matrix3f S = H * P * H.transpose() + R;
//   Eigen::Matrix<float, 6, 3> K = P * H.transpose() * S.inverse();
//   x = x + K * y;
//   P = (Eigen::Matrix<float, 6, 6>::Identity() - K * H) * P;
// }

void FusionImuRadarNode::ekfRadarUpdate(const Eigen::Vector3f &v_meas, Eigen::Matrix<float, 6, 1> &x, Eigen::Matrix<float, 6, 6> &P, const Eigen::Matrix3f &R)
{
    // H 행렬을 위치가 아닌 속도(3,4,5번 인덱스)를 가리키게 수정
    Eigen::Matrix<float, 3, 6> H = Eigen::Matrix<float, 3, 6>::Zero();
    H(0, 3) = 1.0f; // vx
    H(1, 4) = 1.0f; // vy
    H(2, 5) = 1.0f; // vz

    Eigen::Vector3f v_pred = H * x;
    Eigen::Vector3f y = v_meas - v_pred; // 속도 오차

    Eigen::Matrix3f S = H * P * H.transpose() + R;
    Eigen::Matrix<float, 6, 3> K = P * H.transpose() * S.inverse();

    x = x + K * y;
    P = (Eigen::Matrix<float, 6, 6>::Identity() - K * H) * P;
}