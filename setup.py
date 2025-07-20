import os
import subprocess
from setuptools import setup, find_packages

def _process_requirements():
    if not os.path.exists('requirements.txt'):
        return []  # 如果文件不存在，返回空列表

    with open('requirements.txt', 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            result = subprocess.run(['pip', 'install', pkg], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install {pkg}: {result.stderr}")
        else:
            requires.append(pkg)
    
    return requires

setup(
    name="hhcodec",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=_process_requirements(),
)
