from setuptools import setup

package_name = 'puzzlebot_line'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='puzzlebot',
    maintainer_email='puzzlebot@todo.todo',
    description='Puzzlebot line follower with YOLO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_yolo_viva = puzzlebot_line.traffic_line:main',
        ],
    },
)
