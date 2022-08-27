from setuptools import find_packages, setup
setup(
    name='mindbox',
    packages=find_packages(include=['mindbox']),
    version='0.1.0',
    description='A minimal, beginner-friendly Neural Networks library 🤖',
    author='appvoid',
    license='MIT',
    install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==7.1.2'],
    test_suite='tests',
)