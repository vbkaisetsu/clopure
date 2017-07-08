#!/usr/bin/env python3

from setuptools import setup


setup(
    name = "clopure",
    version = "0.1",
    description = "Multi-processing supported functional language for Python",
    url = "https://github.com/vbkaisetsu/clopure",
    author = "Koichi Akabe",
    author_email = "vbkaisetsu@gmail.com",
    license = "MIT",
    packages = ["clopure"],
    test_suite = "test",
    scripts = ["bin/clopure"],
)
