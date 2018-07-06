# -*- coding: utf-8 -*-
# @Time    : 7/6/18 21:18
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="taskbot",
    version="0.0.1",
    author="Zhirui Zhou",
    author_email="evilpsycho42@gmail.com",
    description="a task oriented chatbot framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EvilPsyCHo/TaskBot",
    # packages=setuptools.find_packages(),
    packages=["taskbot"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux"
    ],
)
