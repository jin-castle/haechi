from setuptools import setup

package_name = 'thermal_camera'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DS JUNG',
    maintainer_email='edison041812@gmail.com',
    description='ROS 2 node for thermal camera',
    license='Apache License 2.0',
    tests_require=['pytest'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/thermal_camera']),
        ('share/thermal_camera', ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'thermal_camera_node = thermal_camera.thermal_camera_node:main',
        ],
    },
)

