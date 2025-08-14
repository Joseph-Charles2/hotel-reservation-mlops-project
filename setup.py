from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.strip() for req in requirements if req.strip() != HYPEN_E_DOT]
    return requirements

setup(
    name="MLOPS-Project-1",
    version="0.0.1",
    author="Joseph Charles",
    author_email="josephcharles.19988@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
