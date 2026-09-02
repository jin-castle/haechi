import math
import os
import sys
import time
from fcntl import ioctl

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


G_TO_MS2 = 9.80665
DEG2RAD = math.pi / 180.0

MPU6050_DEFAULT_ADDR = 0x68
MPU6050_PWR_MGMT_1 = 0x6B
MPU6050_ACCEL_XOUT_H = 0x3B
MPU6050_GYRO_XOUT_H = 0x43
MPU6050_ACCEL_LSB_PER_G_2G = 16384.0
MPU6050_GYRO_LSB_PER_DPS_250DPS = 131.0

LSM6DSV16X_DEFAULT_ADDR = 0x6A
LSM6DSV16X_WHO_AM_I = 0x0F
LSM6DSV16X_WHO_AM_I_VALUE = 0x70
LSM6DSV16X_CTRL1 = 0x10
LSM6DSV16X_CTRL2 = 0x11
LSM6DSV16X_CTRL3 = 0x12
LSM6DSV16X_CTRL6 = 0x15
LSM6DSV16X_OUTX_L_G = 0x22
LSM6DSV16X_OUTX_L_A = 0x28
LSM6DSV16X_ACCEL_2G_MG_PER_LSB = 0.061
LSM6DSV16X_GYRO_125DPS_MDPS_PER_LSB = 4.375
I2C_SLAVE = 0x0703


def _signed_16(low, high):
    value = (high << 8) | low
    if value & 0x8000:
        value -= 65536
    return value


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

    def write_byte_data(self, address, register, value):
        self._select(address)
        os.write(self.fd, bytes([register & 0xFF, value & 0xFF]))

    def read_i2c_block_data(self, address, register, length):
        self._select(address)
        os.write(self.fd, bytes([register & 0xFF]))
        return list(os.read(self.fd, length))

    def close(self):
        os.close(self.fd)


class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        self.declare_parameter('driver', 'mock')
        self.declare_parameter('i2c_bus', 7)
        self.declare_parameter('i2c_address', 0)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('complementary_alpha', 0.98)

        self.driver = self.get_parameter('driver').get_parameter_value().string_value.lower()
        self.i2c_bus = self.get_parameter('i2c_bus').get_parameter_value().integer_value
        self.i2c_address = self.get_parameter('i2c_address').get_parameter_value().integer_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.alpha = self.get_parameter('complementary_alpha').get_parameter_value().double_value

        self.bus = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = self.get_clock().now()

        self._init_driver()

        self.imu_pub = self.create_publisher(Imu, '/imu', qos_profile_sensor_data)
        self.tf_broadcaster = TransformBroadcaster(self)

        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

    def _init_driver(self):
        if self.driver == 'mock':
            self.get_logger().warning(
                'IMU driver=mock: publishing synthetic stationary data, not real sensor measurements.'
            )
            return

        device_path = "/dev/i2c-%d" % self.i2c_bus
        if not os.path.exists(device_path):
            raise RuntimeError("%s is missing; expose I2C or run this node on the sensor host" % device_path)
        smbus = self._load_smbus_module()

        if self.driver == 'mpu6050':
            if self.i2c_address == 0:
                self.i2c_address = MPU6050_DEFAULT_ADDR
            self.bus = smbus.SMBus(self.i2c_bus)
            time.sleep(0.1)
            self.bus.write_byte_data(self.i2c_address, MPU6050_PWR_MGMT_1, 0)
            time.sleep(0.1)
            self.get_logger().info(
                'MPU6050 initialized on I2C bus %d, address 0x%02X'
                % (self.i2c_bus, self.i2c_address)
            )
            return

        if self.driver == 'lsm6dsv16x':
            if self.i2c_address == 0:
                self.i2c_address = LSM6DSV16X_DEFAULT_ADDR
            self.bus = smbus.SMBus(self.i2c_bus)
            who_am_i = self.bus.read_byte_data(self.i2c_address, LSM6DSV16X_WHO_AM_I)
            if who_am_i != LSM6DSV16X_WHO_AM_I_VALUE:
                raise RuntimeError(
                    'LSM6DSV16X WHO_AM_I mismatch at 0x%02X: got 0x%02X, expected 0x%02X'
                    % (self.i2c_address, who_am_i, LSM6DSV16X_WHO_AM_I_VALUE)
                )

            self.bus.write_byte_data(self.i2c_address, LSM6DSV16X_CTRL3, 0x44)
            self.bus.write_byte_data(self.i2c_address, LSM6DSV16X_CTRL6, 0x00)
            self.bus.write_byte_data(self.i2c_address, LSM6DSV16X_CTRL1, 0x06)
            self.bus.write_byte_data(self.i2c_address, LSM6DSV16X_CTRL2, 0x06)
            time.sleep(0.1)
            self.get_logger().info(
                'LSM6DSV16X initialized on I2C bus %d, address 0x%02X'
                % (self.i2c_bus, self.i2c_address)
            )
            return

        raise ValueError(
            "Unsupported IMU driver '%s'. Use mock, mpu6050, or lsm6dsv16x." % self.driver
        )

    def _read_mpu6050_word(self, reg):
        high = self.bus.read_byte_data(self.i2c_address, reg)
        low = self.bus.read_byte_data(self.i2c_address, reg + 1)
        return _signed_16(low, high)

    @staticmethod
    def _load_smbus_module():
        try:
            import smbus

            return smbus
        except ImportError:
            try:
                import smbus2

                return smbus2
            except ImportError:
                class LinuxI2CModule:
                    SMBus = LinuxI2CBus

                return LinuxI2CModule

    def _read_mock(self):
        return (0.0, 0.0, G_TO_MS2), (0.0, 0.0, 0.0)

    def _read_mpu6050(self):
        acc_x_raw = self._read_mpu6050_word(MPU6050_ACCEL_XOUT_H)
        acc_y_raw = self._read_mpu6050_word(MPU6050_ACCEL_XOUT_H + 2)
        acc_z_raw = self._read_mpu6050_word(MPU6050_ACCEL_XOUT_H + 4)
        gyro_x_raw = self._read_mpu6050_word(MPU6050_GYRO_XOUT_H)
        gyro_y_raw = self._read_mpu6050_word(MPU6050_GYRO_XOUT_H + 2)
        gyro_z_raw = self._read_mpu6050_word(MPU6050_GYRO_XOUT_H + 4)

        accel = (
            acc_x_raw / MPU6050_ACCEL_LSB_PER_G_2G * G_TO_MS2,
            acc_y_raw / MPU6050_ACCEL_LSB_PER_G_2G * G_TO_MS2,
            acc_z_raw / MPU6050_ACCEL_LSB_PER_G_2G * G_TO_MS2,
        )
        gyro = (
            gyro_x_raw / MPU6050_GYRO_LSB_PER_DPS_250DPS * DEG2RAD,
            gyro_y_raw / MPU6050_GYRO_LSB_PER_DPS_250DPS * DEG2RAD,
            gyro_z_raw / MPU6050_GYRO_LSB_PER_DPS_250DPS * DEG2RAD,
        )
        return accel, gyro

    def _read_lsm6dsv16x(self):
        gyro_bytes = self.bus.read_i2c_block_data(self.i2c_address, LSM6DSV16X_OUTX_L_G, 6)
        accel_bytes = self.bus.read_i2c_block_data(self.i2c_address, LSM6DSV16X_OUTX_L_A, 6)

        gyro_raw = (
            _signed_16(gyro_bytes[0], gyro_bytes[1]),
            _signed_16(gyro_bytes[2], gyro_bytes[3]),
            _signed_16(gyro_bytes[4], gyro_bytes[5]),
        )
        accel_raw = (
            _signed_16(accel_bytes[0], accel_bytes[1]),
            _signed_16(accel_bytes[2], accel_bytes[3]),
            _signed_16(accel_bytes[4], accel_bytes[5]),
        )

        accel = tuple(raw * LSM6DSV16X_ACCEL_2G_MG_PER_LSB / 1000.0 * G_TO_MS2 for raw in accel_raw)
        gyro = tuple(raw * LSM6DSV16X_GYRO_125DPS_MDPS_PER_LSB / 1000.0 * DEG2RAD for raw in gyro_raw)
        return accel, gyro

    def _read_sensor(self):
        if self.driver == 'mock':
            return self._read_mock()
        if self.driver == 'mpu6050':
            return self._read_mpu6050()
        if self.driver == 'lsm6dsv16x':
            return self._read_lsm6dsv16x()
        raise RuntimeError("Unsupported IMU driver '%s'" % self.driver)

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now.nanoseconds - self.last_time.nanoseconds) / 1e9
        if dt <= 0.0:
            dt = 1e-3
        self.last_time = now

        (acc_x, acc_y, acc_z), (gyro_x, gyro_y, gyro_z) = self._read_sensor()

        roll_acc = math.atan2(acc_y, acc_z)
        pitch_acc = math.atan2(-acc_x, math.sqrt(acc_y * acc_y + acc_z * acc_z))

        roll_gyro = self.roll + gyro_x * dt
        pitch_gyro = self.pitch + gyro_y * dt
        yaw_gyro = self.yaw + gyro_z * dt

        self.roll = self.alpha * roll_gyro + (1.0 - self.alpha) * roll_acc
        self.pitch = self.alpha * pitch_gyro + (1.0 - self.alpha) * pitch_acc
        self.yaw = yaw_gyro

        qx, qy, qz, qw = self.rpy_to_quaternion(self.roll, self.pitch, self.yaw)

        imu_msg = Imu()
        imu_msg.header.stamp = now.to_msg()
        imu_msg.header.frame_id = self.frame_id
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.orientation_covariance[0] = 0.01
        imu_msg.orientation_covariance[4] = 0.01
        imu_msg.orientation_covariance[8] = 0.01
        imu_msg.angular_velocity.x = gyro_x
        imu_msg.angular_velocity.y = gyro_y
        imu_msg.angular_velocity.z = gyro_z
        imu_msg.angular_velocity_covariance[0] = 0.02
        imu_msg.angular_velocity_covariance[4] = 0.02
        imu_msg.angular_velocity_covariance[8] = 0.02
        imu_msg.linear_acceleration.x = acc_x
        imu_msg.linear_acceleration.y = acc_y
        imu_msg.linear_acceleration.z = acc_z
        imu_msg.linear_acceleration_covariance[0] = 0.04
        imu_msg.linear_acceleration_covariance[4] = 0.04
        imu_msg.linear_acceleration_covariance[8] = 0.04
        self.imu_pub.publish(imu_msg)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = self.frame_id
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = imu_msg.orientation
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def rpy_to_quaternion(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def destroy_node(self):
        if self.bus is not None:
            self.bus.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImuNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info('Keyboard interrupt, shutting down IMU node.')
    except Exception as exc:
        print("IMU startup failed: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
