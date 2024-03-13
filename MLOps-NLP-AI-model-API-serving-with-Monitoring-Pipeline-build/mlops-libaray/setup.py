from setuptools import setup

setup (
    name="mlopslib",
    version="0.0.1",
    description="custom lib for mlops",
    url="https://gitlab.com/kiru6923110/mlops-libaray",# gitlab url 의미
    author="kiru",
    packages=['mlopslib'],# package에 포함될 디렉토리
    install_requires=[
        "google-cloud-storage==2.6.0"
        ],
)