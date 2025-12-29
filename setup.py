"""Setup script for Terminal Guidance."""

from setuptools import setup, find_packages

setup(
    name="terminal-guidance",
    version="0.1.0",
    description="Drone companion computer for target tracking",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "PyYAML>=6.0",
        "ultralytics>=8.0.0",
        "pymavlink>=2.4.40",
        "scipy>=1.11.0",
        "flask>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "terminal-guidance=app:main",
        ],
    },
)
