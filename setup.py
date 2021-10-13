from setuptools import find_packages, setup
setup(
    name='jacaranda',
    packages=find_packages(include=['jacaranda']),
    version='0.1.0',
    description='Machine Learning Hyperparmater Tuning Pipeline',
    author='Gabriel Dennis, Harry Goodman',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)