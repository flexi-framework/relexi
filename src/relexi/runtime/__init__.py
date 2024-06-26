"""Provides functionalities to create and manage an HPC runtime environment.

This module provides the necessary functionalities to create and manage a
runtime environment on distributed HPC systems for distributed Reinforcement
Learning (RL) algorithms. The main class is `Runtime`, which is used to
identify the resources available on the system, create the necessary
environment variables, and run the given program. The `helpers` module provides
some helper functions to facilitate the process of creating and managing the
runtime environment.

The public classes and functions are:
    - `Runtime`: The main class to create and manage a runtime environment.
    - `helpers`: A module with helper functions to facilitate the process.
"""
from .runtime import Runtime
from . import helpers

__all__ = ['Runtime', 'helpers']
