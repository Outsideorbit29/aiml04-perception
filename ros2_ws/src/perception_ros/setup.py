from glob import glob

from setuptools import setup

PACKAGE = "perception_ros"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aiml04",
    maintainer_email="noreply@example.com",
    description="AIML-04 ROS2 nodes (thin wrappers around perception/ and sim/)",
    license="MIT",
    entry_points={
        "console_scripts": [
            f"carla_bridge_node = {PACKAGE}.carla_bridge_node:main",
            f"perception_node = {PACKAGE}.perception_node:main",
            f"control_node = {PACKAGE}.control_node:main",
            f"recorder_node = {PACKAGE}.recorder_node:main",
        ],
    },
)
