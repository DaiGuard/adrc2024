import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'adrc_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DaiGuard',
    maintainer_email='zenith.or.w@gmail.com',
    description='auto drive RC ros package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_capture = adrc_ros.data_capture:main',
            'test_run = adrc_ros.test_run:main',
        ],
    },
)
