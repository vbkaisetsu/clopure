#!/usr/bin/env python3

from setuptools import setup


setup(
    name = "clopure",
    version = "0.1.3",
    description = "Multi-processing supported functional language",
    url = "https://github.com/vbkaisetsu/clopure",
    author = "Koichi Akabe",
    author_email = "vbkaisetsu@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Interpreters",
    ],
    license = "MIT",
    packages = ["clopure"],
    test_suite = "test",
    scripts = ["bin/clopure"],
)
