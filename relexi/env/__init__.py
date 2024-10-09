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


"""Entails predefined environments for Relexi.

All environments (i.e. subpackages) entailed in this package are added
dynamically to its public API if and only if they entail a `config.yaml` file.

TODO:
    Add validation step to ensure that `config.yaml` is indeed a valid
        configuration file containing all necessary information.
"""

import os
import pkgutil
import inspect
import importlib

from tf_agents.environments import py_environment


def get_environments(my_dir):
    """Get all environments in a directory.

    Prepares list of available Relexi environments in a given directory.
    For this, it iterates over all subdirectories and checks if they contain
    a `config.yaml` file. If so, the subdirectory is added to the list of
    available environments.

    Args:
        my_dir (str): Directory to search for environments.

    Returns:
        list: List of all environments in the directory.
    """
    envs = []
    for _, package_name, is_pkg in pkgutil.iter_modules([my_dir]):
        package_path = os.path.join(my_dir, package_name)
        if is_pkg and os.path.exists(os.path.join(package_path, 'config.yaml')):
            envs.append(package_name)
    return envs


# Get all available environments
envs = get_environments(os.path.dirname(__file__))
"""list: List of all available environments in the package."""
# Add all routines and environments to public API
__all__ = ['get_environments', 'load_environment'] + envs


def load_environment(name):
    """Load environment by name.

    Args:
        name (str): Name of the environment to load.

    Returns:
        class: The class for the requested module.
    """
    if not name in envs:
        raise ValueError(f"Environment '{name}' not found in {__name__}.")

    # First load the module
    try:
        env = importlib.import_module('relexi.env.'+name)
    except ImportError as e:
        raise ImportError(f"Environment '{name}' not found in {__name__}.") from e

    # Find actual environment class in the module
    PARENT_CLASS = py_environment.PyEnvironment
    env_class = []
    for _, module_name, _ in pkgutil.walk_packages(env.__path__, env.__name__ + "."):
        try:
            # Import the module
            module = importlib.import_module(module_name)
            # Iterate over all members of the module
            for name, obj in inspect.getmembers(module):
                # Check if a class and a subclass of parent class
                if inspect.isclass(obj) and issubclass(obj, PARENT_CLASS) and obj is not PARENT_CLASS:
                    env_class.append(obj)
        except ImportError:
            raise ImportError(f"Error while loading environment '{name}'.") from e

    # Check if only unique class was found
    if len(env_class) != 1:
        raise ValueError(f"Found {len(env_class)} classes in environment '{name}', expected exactly one.")

    # Return the class
    return env_class[0]
