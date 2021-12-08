from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

description = """Multiagent Reinforcement Learning Lib + Ray support"""

setup(
    name="VALUE_MIX",
    version="0.0.1",
    description="Multiagent Reinforcement Learning Lib + Ray support",
    long_description=description,
    author="wwx",
    author_email="wxwang@tju.edu.cn",
    license="MIT License",
    keywords="MARL lib + ray",
    packages=[
        "value_mix",
        "value_mix.mixers",
        "value_mix.configs"
    ]
)