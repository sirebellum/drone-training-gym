import setuptools

long_description = 'tetris gym environment wrapped to run on ezspark'

setuptools.setup(
    name = 'path_finder',
    version = '0.0.1',
    install_requires = ['gym', 'pybullet', 'numpy'],
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    description = long_description,
    url = 'https://github.com/ez-spark/gym-mytetris',
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
)
