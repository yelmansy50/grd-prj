from setuptools import find_packages, setup
from typing import List

h = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''this function to get the requirements from the file'''
    requirements = []
    with open(file_path) as f:
        requirements = f.read().splitlines()
        if h in requirements:
            requirements.remove(h)
    return requirements

setup(
    name='mlproject',
    version='0.0.12',  # Corrected version number
    author='elmesery77',
    
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)