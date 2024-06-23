from setuptools import setup, find_packages

setup(
    name='torchobserver',
    version='0.0.0',
    packages=find_packages(include=["torchobserver"]),
    install_requires=[
        # List any dependencies your package requires
        'torch',
        'torchaudio',
        'matplotlib',
        'wandb',
        'soundfile',
    ],
)