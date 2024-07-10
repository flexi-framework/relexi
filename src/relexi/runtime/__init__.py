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
