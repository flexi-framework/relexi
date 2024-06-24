#!/usr/bin/env python3

"""The ClusterManager class for managing the HPC environment."""

import os
import socket
from typing import List, Optional

from smartsim import Experiment
from smartsim.database.orchestrator import Orchestrator

import relexi.io.output as rlxout


class ClusterManager:
    """Class containing information about and handling the HPC environment.

    This class defines the interface for cluster managers, which contain all
    information of the HPC environment used for the training environments and
    method to manage it. This include in particular to identify the scheduler
    environment, identify the hostnames of the available nodes as well as
    launching and managing the SmartSim experiment including the Orchestrator.

    Two possible modes are available for using the available compute resources:

        - **Distributed Mode**: The localhost running the main training script
            becomes the dedicated Head node that hosts the database and runs
            the model evaluation and training loop. All training environments
            are distributed to the available Worker nodes.

        - **Local Mode**: The training script, the database and the training
            environments are all placed on the localhost.

    For **Distributed Mode**, more than 1 node has to be available. Otherwise
    **Local Mode** will be used. The mode of the `ClusterManager` can be
    retrieved via the `is_distributed` attribute.

    Attributes:
        type (str): Type of the cluster manager. Must be one of
            **{'local', 'pbs', 'slurm'}**.
        is_distributed (bool): Whether dedicated head and worker nodes are
            available or everything runs on single shared node.
        hosts (list): List of hostnames.
        head (str): Hostname of Head node.
        workers (list): List of worker nodes.

    Methods:
        print_info: Print information about the current environment.
        generate_rankfiles: Generate rank files for OpenMPI process binding.

    Raises:
        ValueError: If the scheduler type is not supported.
        RuntimeError: If the following conditions are met:
            - The scheduler environment cannot be identified, or
            - Launching the Orchestrator fails.
        NotImplementedError: If the methods are not implemented for the
            provided scheduler type.
    """

    TYPES = ['local', 'pbs', 'slurm']  # All implemented types of cluster managers

    def __init__(
            self,
            scheduler_type: Optional[str] = 'local',
            db_port: Optional[int] = 6790,
            db_network_interface: Optional[str] = 'lo'
            ):
        self.type = scheduler_type.casefold().strip()
        rlxout.info(f'Trying to identify {self.type} training environment...')

        try:
            self._hosts = self._get_hostlist()
        except Exception as e:
            rlxout.warning(f'Failed: {e}')
            if self.type != 'local':
                rlxout.info('Trying to run in local mode instead...', newline=False)
                try:
                    self.type = 'local'
                    self._hosts = self._get_hostlist()
                except Exception as e:
                    raise RuntimeError('Also failed to setup local environment!') from e
            else:
                raise RuntimeError('Failed to setup local training environment!') from e
        rlxout.info('Success!', newline=False)

        self.db = None
        self.exp, self.db, self.entry_db = self._launch_orchestrator(
            port=db_port,
            network_interface=db_network_interface,
        )

    def __del__(self):
        if self.db:
            try:
                self.exp.stop(self.db)
            except Exception as e:
                raise RuntimeError('Failed to stop the Orchestrator!') from e

    def _launch_orchestrator(
            self,
            port: int,
            network_interface: str
            ) -> tuple[Experiment, Orchestrator, str]:
        """Launches the SmartSim Orchestrator for the current job.

        Args:
            port (int): Port to start the Orchestrator on.
            network_interface (str): Network interface to use for the Orchestrator.

        Returns:
            tuple: The Experiment object, the Orchestrator object, and the entry database hostname.
        """
        rlxout.small_banner('Starting Orchestrator...')

        # Generate flexi experiment
        exp = Experiment('flexi', launcher=self.type)

        # Initialize the orchestrator based on the orchestrator_type
        db = exp.create_database(
            port=port,
            interface='lo' if self.type == 'local' else network_interface,
            hosts=self.head if self.type in {'pbs', 'slurm'} else None,
        )

        rlxout.info("Starting the Database...", newline=False)
        try:
            exp.start(db)
        except Exception as e:
            raise RuntimeError(f"Failed to start the Orchestrator: {e}") from e
        rlxout.info("Success!", newline=False)

        entry_db = socket.gethostbyname(db.hosts[0])
        rlxout.info("If the SmartRedis database isn't stopping properly you can use this command to stop it from the command line:")
        rlxout.info(f"$(smart dbcli) -h {db.hosts[0]} -p {port} shutdown", newline=False)

        return exp, db, entry_db

    def info(self):
        """Print information about the current environment."""
        rlxout.info("Found the following environment:")
        rlxout.info(f"  Scheduler: {self.type}", newline=False)
        rlxout.info(f"  Hosts:     {self.hosts}", newline=False)
        if self.is_distributed:
            rlxout.info(f"Relexi is running in distributed mode:")
            rlxout.info(f"  Head node: {self.head}", newline=False)
            rlxout.info(f"  Workers:   {self.workers}", newline=False)
        else:
            rlxout.info(f"Relexi is running in local mode on: {self.head}")

    def _get_hostlist(self) -> List[str]:
        """Get the list of hosts the script is executed on.

        Uses the scheduler type to determine the hostlist via the environment
        variables set by the scheduler.

        Returns:
            list: List containing the hostnames as strings.

        Raises:
            NotImplementedError: If the method is not implemented for the
                scheduler type.
        """
        if self.type == 'local':
            return [socket.gethostname()]
        elif self.type == 'pbs':
            return os.environ['PBS_NODEFILE']
        elif self.type == 'slurm':
            return os.environ['SLURM_NODELIST']
        else:
            raise NotImplementedError(f"Method get_hostlist not implemented for scheduler type {self.type}")

    @property
    def type(self) -> str:
        """Get the type of scheduler environment used for the cluster manager.

        Returns:
            str: Type of the cluster manager.
        """
        return self._type

    @type.setter
    def type(self, value: str):
        """Set the type of scheduler environment used for the cluster manager.
        Ensure that the type is supported.

        Args:
            value (str): Type of the cluster manager.
        """
        if value not in self.TYPES:
            raise ValueError(f"Scheduler type {value} not supported.")
        self._type = value

    @property
    def hosts(self) -> List[str]:
        """Get the list of hosts the script is executed on.

        Returns:
            list: List containing the hostnames as strings.
        """
        return self._hosts

    @property
    def is_distributed(self) -> bool:
        """Returns whether ClusterManager runs in distributed or local mode.

        Returns:
            bool: Indicates whether cluster runs in distributed or local mode.
                - True: A dedicated head node is used for training and the
                    database and at least one additional worker node to run
                    simulations.
                - False: Only single machine is available and training,
                    database and simulation will be performed on the localhost.
        """
        return len(self._hosts) > 1

    @property
    def head(self) -> str:
        """Get head node, which is where Relexi is actually runs on.

        Returns:
            str: Hostname of the head node.
        """
        return self.hosts[0]

    @property
    def workers(self) -> List[str]:
        """Get a list of worker nodes depending on mode.

        Returns:
            list: List containing the hostnames of worker nodes as strings.
        """
        if self.is_distributed:
            return self.hosts[1:]
        return self.hosts

    def generate_rankfiles(self, n_models: int, n_ranks_per_model: int, base_path: Optional[str] = None) -> List[str]:
        """Generate rank file for OpenMPI process binding.

        Args:
            n_models (int): Number of models to be launched.
            n_ranks_per_model (int): Number of ranks used for each model.
            base_path (str, optional): Path to the directory of the rank files.

        Returns:
            list: List of filenames of the rankfiles.
        """
        if self.type not in self.TYPES:
            raise NotImplementedError(f"Method generate_rankfile not implemented for scheduler type {self.type}")
        return generate_rankfile_ompi(self.hosts, n_models, n_ranks_per_model, base_path)
