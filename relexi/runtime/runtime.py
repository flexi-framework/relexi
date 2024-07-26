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


"""The Runtime class for managing the HPC runtime environment."""

import os
import socket
import subprocess
from typing import List, Optional, Union

import numpy as np

import smartsim
from smartsim import Experiment
from smartsim.database.orchestrator import Orchestrator

import relexi.io.output as rlxout
from relexi.runtime import LaunchConfig


class Runtime():
    """Class containing information about and handling the HPC environment.

    This class defines the interface for an HPC runtime, which contains all
    information of the HPC environment used for the training environments and
    methods to manage it. This includes in particular to identify the scheduler
    environment, the hostnames of the available nodes and launching and
    managing the SmartSim `Experiment` including the `Orchestrator`.

    Two possible modes are available for using the available compute resources:

        - **Distributed**: The `localhost` running the main training script
            becomes the dedicated **Head** node that hosts the database,
            evaluates the model and runs the training loop. All training
            environments are distributed to the available **Worker** nodes.

        - **Local**: The training script, the database and the training
            environments are all placed on the `localhost`.

    More than 1 node has to be available in order to initiate a
    **Distributed** runtime. Otherwise, if only a single node is available,
    **Local** mode will be used. The mode of the `Runtime` can be retrieved
    via the `is_distributed` attribute.

    Attributes:
        type (str): Type of runtime. Must be `'local'`, `'pbs'`, or `'slurm'`.
        is_distributed (bool): Indicates whether a **Distributed** or **Local**
            runtime is used.
        hosts (list): List of hostnames of available nodes.
        head (str): Hostname of Head node (is name of `localhost` if in
            **Local Mode**).
        workers (list): List of worker nodes (contains only `localhost` if in
            **Local Mode**).
        n_worker_slots (int): Total number of slots available on workers.
        db (Orchestrator): The launched `Orchestrator` database from the
            `smartsim` packagev.
        db_entry (str): IP address and port of the host of the database.
            Takes the form `IP_ADDRESS:PORT`.
        exp (Experiment): The `Experiment` object the `Orchestrator` is
            launched with.
        launch_config (LaunchConfig): CurrentcConfiguration for launching a
            batch of executables in the runtime.

    Raises:
        ValueError: If the scheduler type is not supported.
        RuntimeError: If the following conditions are met:
            - The scheduler environment cannot be identified, or
            - Launching the `Orchestrator` fails.
        NotImplementedError: If the methods are not implemented for the
            provided scheduler type.
    """

    TYPES = ['local', 'pbs', 'slurm']
    """Supported types of runtime environments."""

    def __init__(
            self,
            type_: Optional[str] = 'auto',
            db_network_interface: Optional[str] = 'lo',
            db_port: Optional[int] = 6790,
            do_launch_orchestrator: Optional[bool] = True
            ):
        """Initialize the Runtime.

        Args:
            type_ (str, optional): Type of runtime. Must be `'local'`, `'pbs'`,
                `'slurm'` or `'auto'`. Defaults to `'auto'`, for which the type
                of runtime environment is identified automatically.
            db_network_interface (str, optional): Network interface to use for
                the Orchestrator. Defaults to `'lo'`.
            db_port (int, optional): Port to start the Orchestrator on.
                Defaults to `6790`.
            do_launch_orchestrator (bool, optional): Whether to launch the
                `Orchestrator` immediately. Defaults to `True`.
        """
        try:
            # Using SmartSim utility to identify type automatically
            if type_ == 'auto':
                rlxout.info('Identifying environment...')
                scheduler = smartsim.wlm.detect_launcher()
                rlxout.info(f'Found "{scheduler}" environment!', newline=False)
                self.type = scheduler.casefold().strip()
            else:
                self.type = type_.casefold().strip()

            rlxout.info(f'Setting up "{self.type}" runtime...')
            self._hosts = self._get_hostlist()
            # Check that actually sufficient hosts found
            if self.type != 'local' and len(self._hosts) < 2:
                raise ValueError('Less than 2 hosts found in environment!')
        except Exception as e:
            rlxout.warning(f'Failed: {e}')
            if type_ != 'local':
                rlxout.info('Trying to setup LOCAL runtime instead...', newline=False)
                try:
                    self.type = 'local'
                    self._hosts = self._get_hostlist()
                except Exception as f:
                    raise RuntimeError('Also failed to setup LOCAL environment!') from f
            else:
                raise RuntimeError('Failed to setup LOCAL training environment!') from e
        rlxout.info('Success!', newline=False)

        self._exp = None
        self._db = None
        self._db_entry = None
        if do_launch_orchestrator:
            try:
                self._exp, self._db, self._db_entry = self._launch_orchestrator(
                    port=db_port,
                    network_interface=db_network_interface,
                )
            except Exception as e:
                raise RuntimeError('Failed to launch the Orchestrator!') from e
        self.launch_config = None
        self.n_worker_slots = self._get_total_worker_slots()

    def __del__(self):
        if self.db:
            try:
                self.exp.stop(self.db)
            except Exception as e:
                raise RuntimeError('Failed to stop the Orchestrator!') from e

    def info(self):
        """Prints information about the current runtime environment."""
        rlxout.info('Configuration of runtime environment:')
        rlxout.info(f'  Scheduler: {self.type}', newline=False)
        rlxout.info(f'  Hosts:     {self.hosts}', newline=False)
        if self.is_distributed:
            rlxout.info('Running in DISTRIBUTED mode:')
            rlxout.info(f'  Head:      {self.head}', newline=False)
            rlxout.info(f'  Workers:   {self.workers}', newline=False)
        else:
            rlxout.info(f'Running in LOCAL mode on: {self.head}')

    def launch_models(
            self,
            exe: Union[str, List[str]],
            exe_args: Union[str, List[str]],
            exe_name: Union[str, List[str]],
            n_procs: Union[int, List[int]],
            n_exe: Optional[int] = 1,
            launcher: Optional[str] = 'local'
            ) -> List[smartsim.entity.model.Model]:
        """Launch the models on the available nodes.

        Args:
            exe (str, List(str)): Path to the executable to launch. Can either
                be a single path or a list of length `n_exe`. If only a single
                path is provided, it is used for all executables.
            exe_args (str, List(str)): Arguments to pass to the executable. Can
                either be a single string or a list of length `n_exe`. If only
                a single string is provided, it is used for all executables.
            exe_name (str, List(str)): Name of the executable used to identify
                launched model in the SmartSim context. Can either be a single
                string or a list of length `n_exe`. If only a single string is
                provided, it is used for all executables.
            n_procs (int, List(int)): Number of processes to launch. Can either
                be a single integer or a list of length `n_exe`. If only a
                single integer is provided, it is used for all executables.
            n_exe (int): Number of executables to launch. Defaults to `1`.
            launcher (str): Launcher to use for the executable. Must be one of
                `'mpirun'`, `'srun'`, or `'local'`.
        """
        def _validate_args(arg, n):
            """Validate the length of the arguments."""
            if isinstance(arg, list) and not len(arg) == n:
                raise ValueError(f'Expected {n} entries, but got {len(arg)}!')
            if not isinstance(arg, list):
                return [arg] * n
            return arg

        # Validate that arguments are of correct length
        exe = _validate_args(exe, n_exe)
        exe_args = _validate_args(exe_args, n_exe)
        exe_name = _validate_args(exe_name, n_exe)
        n_procs = _validate_args(n_procs, n_exe)

        # Check compatibility of launcher and scheduler type
        if (launcher == 'local') and (max(n_procs) > 1):
            raise ValueError('Local launcher only supports single process execution!')
        if (launcher == 'srun') and (self.type != 'slurm'):
            raise ValueError('srun launcher only supported for SLURM scheduler!')

        # Check if launch config is up-to-date and create or update if required
        config_dict = {'type': launcher,
                       'n_exe': n_exe,
                       'n_procs': n_procs,
                       'workers': self.workers}
        if self.launch_config is None:
            self.launch_config = LaunchConfig.from_dict(config_dict, self)
        else:
            if not self.launch_config.is_compatible(config_dict):
                self.launch_config.config = config_dict

        models = []
        for i in range(n_exe):
            if launcher == 'local':
                run = self.exp.create_run_settings(
                                                  exe=exe[i],
                                                  exe_args=exe_args[i],
                                                  )
            else:
                if launcher == 'mpirun':
                    run_args = {
                        'rankfile': self.launch_config.rankfiles[i],
                        'report-bindings': None
                    }
                elif launcher == 'srun':
                    run_args = {
                        'mpi': 'pmix',
                        'nodelist': ','.join(self.launch_config.hosts_per_exe[i]),
                        'distribution': 'block:block:block,Pack',
                        'cpu-bind': 'verbose',
                        'exclusive': None,
                    }
                run = self.exp.create_run_settings(
                                                  exe=exe[i],
                                                  exe_args=exe_args[i],
                                                  run_command=launcher,
                                                  run_args=run_args
                                                  )
                run.set_tasks(n_procs[i])

            model = self.exp.create_model(exe_name[i], run)
            self.exp.start(model, block=False, summary=False)
            models.append(model)

        return models

    @property
    def type(self) -> str:
        """Get the type of the runtime environment.

        Returns:
            str: Type of the runtime environment.
        """
        return self._type

    @type.setter
    def type(self, value: str):
        """Set the type of environment used for the runtime.

        Validates that the type is actually supported.

        Args:
            value (str): Type of the runtime environment.
        """
        if value not in self.TYPES:
            raise ValueError(f'Runtime of type {value} not supported.')
        self._type = value

    @property
    def hosts(self) -> List[str]:
        """Get the list of hosts within the runtime environment.

        Returns:
            list: List containing the hostnames as strings.
        """
        return self._hosts

    @property
    def is_distributed(self) -> bool:
        """Whether runtime is **Distributed** or **Local**.

        Checks for the number of hosts available. If more than one host is
        found in runtime, it runs in **Distributed** mode, otherwise it runs
        in **Local** mode.

        Returns:
            bool: `True` if **Distributed**, `False` otherwise.
        """
        return len(self._hosts) > 1

    @property
    def head(self) -> str:
        """Return name of Head node, which is where this instance is located.

        Returns:
            str: Hostname of the Head node.
        """
        return self._get_local_hostname()

    @property
    def workers(self) -> List[str]:
        """Returns list of Workers found in the current runtime environment.

        Obtains Workers by removing the Head node from the list of hosts.

        Returns:
            list: List containing the hostnames of Workers as strings.
        """
        if self.is_distributed:
            workers = self.hosts.copy()
            if self.head in workers:
                workers.remove(self.head)
            else:
                rlxout.warning(f'Localhost "{self.head}" not found in hosts list:')
                rlxout.warning(f'  {workers}')
            return workers
        return self.hosts

    @property
    def db(self) -> Orchestrator:
        """Get the Orchestrator database instance.

        Returns:
            Orchestrator: The `Orchestrator` database instance.
        """
        return self._db

    @property
    def db_entry(self) -> str:
        """Get IP address of database.

        Returns:
            str: Address of the database. Takes the form `IP_ADDRESS:PORT`.
        """
        return self._db_entry

    @property
    def exp(self) -> Experiment:
        """Get the `Experiment` instance the `Orchestrator` is launched in.

        Returns:
            Experiment: The `Experiment` instance.
        """
        return self._exp

    def _launch_orchestrator(
            self,
            port: int,
            network_interface: str
            ) -> tuple[Experiment, Orchestrator, str]:
        """Launches a SmartSim `Orchestratori` in the current runtime.

        Args:
            port (int): Port to start the `Orchestrator` on.
            network_interface (str): Network interface to use for the
                `Orchestrator`.

        Returns:
            tuple: The `Experiment` instance, the `Orchestrator` instance and
                the IP address of the host of the database.
        """
        # Generate relexi experiment
        exp = Experiment('relexi', launcher=self.type)

        # Initialize the orchestrator based on the orchestrator_type
        db = exp.create_database(
            port=port,
            interface='lo' if self.type == 'local' else network_interface,
            hosts=self.head if self.type in {'pbs', 'slurm'} else None,
        )

        rlxout.info('Starting the Orchestrator...', newline=False)
        try:
            exp.start(db)
        except Exception as e:
            raise RuntimeError(f'Failed to start the Orchestrator: {e}') from e
        rlxout.info('Success!', newline=False)

        db_entry = socket.gethostbyname(db.hosts[0])
        rlxout.info('Use this command to shutdown database if not terminated correctly:')
        rlxout.info(f'$(smart dbcli) -h {db.hosts[0]} -p {port} shutdown', newline=False)

        return exp, db, f'{db_entry}:{port}'


    def _get_hostlist(self) -> List[str]:
        """Get the list of hosts the script is executed on.

        Returns:
            list: List containing the hostnames as strings.

        Raises:
            NotImplementedError: If the method is not implemented for the
                scheduler type.
        """
        if self.type == 'local':
            return [self._get_local_hostname()]
        if self.type == 'pbs':
            return smartsim.wlm.pbs.get_hosts()
        if self.type == 'slurm':
            return smartsim.wlm.slurm.get_hosts()
        raise NotImplementedError(
            f'Method `get_hostlist` not implemented for runtime "{self.type}"!')

    def _get_slots_per_node_slurm(self) -> List[int]:
        """Get the number of slots per node for the SLURM scheduler.

        Returns:
            list(int): List containing the number of slots per node.
        """
        if self.type != 'slurm':
            raise ValueError('Method only available for SLURM scheduler!')
        # 1. Get the nodelist
        slots = os.getenv('SLURM_JOB_CPUS_PER_NODE')
        if slots is None:
            raise ValueError("SLURM_JOB_CPUS_PER_NODE is not set!")
        # 2. split all entries at comma
        nodelist = slots.split(',')
        # 3. expand all compressed entries
        expanded_list = []
        for entry in nodelist:
            if '(' in entry:
                num_cpus, count = entry.split('(x')
                num_cpus = int(num_cpus)
                count = int(count[:-1])  # remove trailing ')'
                expanded_list.extend([num_cpus] * count)
            else:
                expanded_list.append(int(entry))
        return expanded_list

    def _get_slots_per_node_pbs(self) -> List[int]:
        """Get the number of slots per node for the PBS scheduler.

        Returns:
            list(int): List containing the number of slots per node.
        """
        if self.type != 'pbs':
            raise ValueError('Method only available for PBS scheduler!')
        # 1. Get the nodelist
        node_file = os.getenv('PBS_NODEFILE')
        if node_file is None:
            raise KeyError('Environment variable "PBS_NODEFILE" not found!')
        # 2. Read the nodelist
        with open(node_file, 'r', encoding='utf-8') as f:
            nodes = [line.strip().split('.')[0] for line in f.readlines()]
        # 3. Count the number of slots (i.e. lines) per node
        return [nodes.count(host) for host in self.hosts]

    def _get_total_worker_slots(self) -> int:
        """Get the total number of worker slots available in the runtime.

        Returns:
            int: Number of slots per worker node.
        """
        if self.type == 'local':
            # Leave one core for the head node
            return os.cpu_count()-1
        if self.type == 'pbs':
            slots_per_node = self._get_slots_per_node_pbs()
            return np.sum(slots_per_node[1:])
        if self.type == 'slurm':
            slots_per_node = self._get_slots_per_node_slurm()
            return np.sum(slots_per_node[1:])
        raise NotImplementedError(
            f'Method `get_slots_per_worker` not implemented for runtime "{self.type}"!')

    def _get_local_hostname(self) -> str:
        """Get the hostname of the machine executing the Python script.

        Returns:
            str: Hostname of the local machine executing the script.
        """
        return socket.gethostname().split('.')[0]
