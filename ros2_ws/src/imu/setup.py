from setuptools import setup

package_name = 'imu'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sumincho',
    maintainer_email='1213tnals@naver.com',
    description='ROS2 IMU publisher for mock, MPU6050, and LSM6DSV16X drivers',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'imu_node = imu.imu_node:main',
        ],
    },
)

