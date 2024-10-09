
from setuptools import setup, find_packages

setup(
    name='GymnasiumRL',
    version='0.1.0',
    description='A Python package for reinforcement learning using Gymnasium.',
    author='Your Name',
    author_email='mkarbasioun@gmail.com',
    url='https://github.com/neurovortex/GymnasiumRL',
    packages=find_packages(include=['gymnasium_rl', 'gymnasium_rl.*']),
    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=2.1.1',
        'matplotlib>=3.9.2',
        'pytest>=8.3.3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Apache License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
