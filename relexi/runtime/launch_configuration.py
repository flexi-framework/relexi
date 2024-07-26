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


"""Launch configuration for a batch of executables in a runtime."""

from __future__ import annotations

from typing import List

import numpy as np

class LaunchConfig():
    """Launch configuration for a batch of executables in a runtime.

    This class provides a launch configuration for a batch of executables in a
    runtime. It contains the specific configuration to distribute the
    executables to the available resources. The configuration can be of three
    types: 'local', 'mpirun', and 'srun'. The 'local' configuration is used
    for local execution, 'mpirun' for OpenMPI, and 'srun' for SLURM.

    Attributes:
        type (str): Type of the launch configuration.
        n_exe (int): Number of executables to launch.
        n_procs (List[int]): Number of processes to launch per executable. Must
            be of length `n_exe`.
        workers (List[str]): List of worker nodes available.
        n_worker_slots (int): Number of available worker slots.
        config (dict): Configuration dictionary.
        rankfiles (List[str]): List of rankfiles if `type=='mpirun'`, is `None`
            otherwise.
        hosts_per_exe (List[List[str]]): List of lists containing the hostnames
            for each executable if `type=='srun'`, is `None` otherwise.

    Methods:
        from_dict(cls, config: dict, runtime: Runtime) -> LaunchConfiguration:
            Instantiate a launch configuration from a configuration dictionary.
        as_dict() -> dict:
            Return the launch configuration as a dictionary.
        is_compatible(config: dict) -> bool:
            Check if other launch configuration is compatible based on dict.
        config_is_valid(config: dict) -> bool:
            Check if the given configuration is valid.

    Raises:
        ValueError: If the requested configuration is invalid.
        RuntimeError: If the configuration cannot be generated.
    """

    TYPES = ['local', 'mpirun', 'srun']
    """Supported types of launch configurations."""

    CONFIG_KEYS = ['type', 'n_exe', 'n_procs', 'workers']
    """Keys for the configuration dictionary."""

    def __init__(self, type_: str, runtime, n_exe: int, n_procs: List[int]):
        """Initialize the launch configuration.

        Args:
            type_ (str): Type of the launch configuration.
            runtime (Runtime): Runtime instance for which launch configuration
                should be generated.
            n_exe (int): Number of executables to launch.
            n_procs (List[int]): Number of processes to launch per executable.
        """
        self.type = type_
        self.n_exe = n_exe
        self.n_procs = n_procs
        self.workers = runtime.workers
        self.n_worker_slots = runtime.n_worker_slots
        # Set with property setter to check for validity
        self.config = {'type': self.type,
                       'n_exe': self.n_exe,
                       'n_procs': self.n_procs,
                       'workers': self.workers}

        # Generate rankfiles for OpenMPI
        self._rankfiles = None
        if self.type == 'mpirun':
            slots_per_node = runtime.n_worker_slots//len(self.workers)
            self._rankfiles = self._generate_rankfile_ompi(self.workers,
                                                           slots_per_node,
                                                           n_exe,
                                                           n_procs)
        # Distribute workers for SLURM
        self._hosts_per_exe = None
        if self.type == 'srun':
            self._hosts_per_exe = self._distribute_workers_slurm(n_procs,
                                                                 n_exe,
                                                                 runtime.workers,
                                                                 runtime.n_worker_slots)

    @property
    def config(self) -> dict:
        """Return the current launch configuration as dict."""
        return self._config

    @config.setter
    def config(self, config: dict):
        """Set a launch configuration."""
        if not self.config_is_valid(config):
            raise ValueError('Invalid configuration dictionary!')
        if sum(config['n_procs']) > self.n_worker_slots:
            raise ValueError('Not enough processes available!')
        self._config = config

    @property
    def type(self) -> str:
        """Return the type of the launch configuration."""
        return self._type

    @type.setter
    def type(self, value):
        """Set the type of the launch configuration."""
        if value not in self.TYPES:
            raise ValueError('Invalid launch configuration type!')
        self._type = value

    @property
    def rankfiles(self) -> List[str]:
        """Return paths to rankfiles for `mpirun` launcher."""
        if self._rankfiles is None:
            raise ValueError('Rankfiles not yet generated!')
        return self._rankfiles

    @property
    def hosts_per_exe(self) -> List[List[str]]:
        """Return the hosts for each executable for `srun`."""
        if self._hosts_per_exe is None:
            raise ValueError('Hosts not yet generated!')
        return self._hosts_per_exe

    def as_dict(self) -> dict:
        """Return the launch configuration as a dictionary."""
        return self._config

    def is_compatible(self, config: dict) -> bool:
        """Check if other launch configuration is compatible based on dict.

        Another launch configuration is compatible if the first `n_exe`
        executables can be launched on the same resources as the first `n_exe`
        executables of the existing launch configuration.

        Args:
            config (dict): Dictionary of the other launch configuration.

        Returns:
            bool: `True` if the configurations are compatible, `False` otherwise.
        """
        if self.type != config['type']:
            return False
        if self.n_exe != config['n_exe']:
            return False
        if self.n_procs != config['n_procs']:
            return False
        return True

    @classmethod
    def from_dict(cls, config: dict, runtime) -> LaunchConfig:
        """Instantiate a launch configuration from a configuration dictionary.

        The dictionary has to take the form of:
            ```
            {
                'type': str,
                'n_exe': int,
                'n_procs': List[int]
                'workers': List[str]
            }
            ```

        Args:
            config (dict): Configuration dictionary.
            runtime (Runtime): Runtime object.

        Returns:
            LaunchConfig: Launch configuration instance.
        """
        if not cls.config_is_valid(config):
            raise ValueError('Invalid configuration dictionary!')
        return cls(config['type'], runtime, config['n_exe'], config['n_procs'])

    @classmethod
    def config_is_valid(cls, config: dict) -> bool:
        """Check if the given configuration is valid.

        The configuration is valid if it contains all necessary keys and the
        values are valid. However, the availability of the resources is not
        checked!

        Args:
            config (dict): Configuration dictionary.

        Returns:
            bool: `True` if the configuration is valid, `False` otherwise.
        """
        if not all(key in config for key in cls.CONFIG_KEYS):
            raise ValueError('Configuration dictionary does not contain all neccessary keys!')
        if config['type'] not in cls.TYPES:
            return False
        if len(config['n_procs']) != config['n_exe']:
            return False
        if len(config['workers']) < 1:
            return False
        if min(config['n_procs']) < 1:
            return False
        if config['n_exe'] < 1:
            return False
        return True

    @staticmethod
    def _distribute_workers_slurm(
                                  n_procs: List[int],
                                  n_exe: int,
                                  workers: List[str],
                                  procs_avail: int
                                  ) -> List[List[str]]:
        """Distribute the executables to the available nodes for SLURM.

        Uses two different strategies to distribute the executables to the
        available nodes. Either multiple executables per node or multiple nodes
        per executable. However, a single executable cannot be placed on parts
        of multiple nodes, since this causes problems with SLURM. Either
        executable spans multiple whole nodes, or single partial node.

        Args:
            n_procs (List[int]): Number of processes to launch per executable.
            n_exe (int): Number of executables to launch.
            workers (List[str]): List of worker nodes available.
            procs_avail (int): Number of available processes.

        Returns:
            List[List[str]]: List of lists containing the hostnames for each
                executable.
        """
        if sum(n_procs) > procs_avail:
            raise RuntimeError('Failed to distribute models to resources!')
        procs_per_worker = procs_avail//len(workers)
        nodes_avail = workers
        slurm_hosts_per_exe = []
        # Either multiple executables per node or multiple nodes per executable
        if max(n_procs) > procs_per_worker:
            # Use whole nodes per executable
            for i in range(n_exe):
                n_nodes_req = int(np.ceil(n_procs[i]/procs_per_worker))
                current_hosts = []
                for _ in range(n_nodes_req):
                    current_hosts.append(nodes_avail.pop(0))
                slurm_hosts_per_exe.append(current_hosts)
        else:
            # Use multiple executables peper
            cores_avail = procs_per_worker
            for i in range(n_exe):
                # Does not fit on remaining slots on node
                if n_procs[i] > cores_avail:
                    if len(nodes_avail) <= 1:
                        raise RuntimeError('Failed to distribute models to resources!')
                    # Take next node
                    nodes_avail.pop(0)
                    cores_avail = procs_per_worker
                cores_avail -= n_procs[i]
                slurm_hosts_per_exe.append([nodes_avail[0]])
        return slurm_hosts_per_exe

    @staticmethod
    def _generate_rankfile_ompi(workers: List[str],
                                n_slots_per_worker: int,
                                n_exe: int,
                                n_procs: List[int],) -> List[str]:
        """Generate rank file for OpenMPI process binding.

        Args:
            workers (list): List of hostnames
            n_exe (int): Number of executables to be launched
            n_procs (int): Number of ranks per environments

        Returns:
            list: List of filenames of the rankfiles
        """
        rankfiles = []
        next_free_slot = 0
        n_cores_used = 0
        for i_exe in range(n_exe):
            filename = f'.env_{i_exe:05d}.txt'
            rankfiles.append(filename)
            with open(filename, 'w', encoding='utf-8') as rankfile:
                for i in range(n_procs[i_exe]):
                    rankfile.write(f'rank {i}={workers[n_cores_used//n_slots_per_worker]} slot={next_free_slot}\n')
                    next_free_slot += 1
                    n_cores_used += 1
                    if next_free_slot >= n_slots_per_worker:
                        next_free_slot = 0
        return rankfiles
