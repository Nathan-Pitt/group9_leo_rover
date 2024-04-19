from setuptools import find_packages, setup

package_name = 'ros2_opencv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mscrobotics2324laptop2',
    maintainer_email='mscrobotics2324laptop2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_sensor_publisher = ros2_opencv.camera_publisher:main',
            'camera_sensor_subscriber = ros2_opencv.camera_subscriber:main',
            'robot_camera_subscriber = ros2_opencv.robot_camera_subscriber:main'
        ],
    },
)
