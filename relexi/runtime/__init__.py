#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Relexi, a reinforcement learning framework for training
# machine learning models in simulations on high-performance computing systems.
#
# Copyright (c) 2022-2024 Marius Kurz, Andrea Beck
#
# Relexi is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Relexi is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Relexi. If not, see <http://www.gnu.org/licenses/>.


"""Provides functionalities to create and manage an HPC runtime environment.

This module provides the necessary functionalities to create and manage a
runtime environment on distributed HPC systems for distributed Reinforcement
Learning (RL) algorithms. The main class is `Runtime`, which is used to
identify the resources available on the system, create the necessary
environment variables, and run the given program. The `LaunchConfig` class
provides a configuration for launching a batch of executables in the runtime.
This include most importantly the distribution of the executables across the
available resources. The `helpers` module provides some helper functions to
facilitate the process of creating and managing the runtime environment.

The public classes and functions are:
    - `Runtime`: The main class to create and manage a runtime environment.
    - `LaunchConfig`: A class to define the launch configuration for a batch of
            executables in a runtime.
    - `helpers`: A module with helper functions to facilitate the process.
"""
from .launch_configuration import LaunchConfig
from .runtime import Runtime
from . import helpers

__all__ = ['Runtime', 'LaunchConfig', 'helpers']
