#!/usr/bin/env python3

"""The ClusterManager class for managing the HPC environment."""

import os
import socket
from typing import List, Optional

from smartsim import Experiment
from smartsim.database.orchestrator import Orchestrator

import relexi.io.output as rlxout
from relexi.smartsim.helpers import generate_rankfile_ompi


class ClusterManager:
    """Class containing information about and handling the HPC environment.

    This class defines the interface for cluster managers, which contain all
    information of the HPC environment used for the training environments and
    methods to manage it. This includes in particular to identify the scheduler
    environment, the hostnames of the available nodes and launching and
    managing the SmartSim `Experiment` including the `Orchestrator`.

    Two possible modes are available for using the available compute resources:

        - **Distributed Mode**: The `localhost` running the main training script
            becomes the dedicated **Head** node that hosts the database,
            evaluates the model and runs the training loop. All training
            environments are distributed to the available **Worker** nodes.

        - **Local Mode**: The training script, the database and the training
            environments are all placed on the `localhost`.

    For **Distributed Mode**, more than 1 node has to be available. Otherwise
    **Local Mode** will be used. The mode of the `ClusterManager` can be
    retrieved via the `is_distributed` attribute.

    Attributes:
        type (str): Type of the cluster manager. Must be `'local'`, `'pbs'`,
            or `'slurm'`.
        is_distributed (bool): Indicates whether cluster runs in
            **Distributed Mode** or **Local Mode**.
        hosts (list): List of hostnames of available nodes.
        head (str): Hostname of Head node (is name of `localhost` if in
            **Local Mode**).
        workers (list): List of worker nodes (contains only `localhost` if in
            **Local Mode**).
        db (Orchestrator): The launched `Orchestrator` database from the
            `smartsim` package.
        exp (Experiment): The `Experiment` object the `Orchestrator` is
            launched with.
        entry_db (str): IP address of the host of the database. Required to
            connect a client to the database.

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
            db_network_interface: Optional[str] = 'lo',
            db_port: Optional[int] = 6790
            ):
        """Initialize the ClusterManager.

        Args:
            scheduler_type (str, optional): Type of the cluster manager.
                Must be `'local'`, `'pbs'`, or `'slurm'`. Defaults to `'local'`.
            db_network_interface (str, optional): Network interface to use for
                the Orchestrator. Defaults to `'lo'`.
            db_port (int, optional): Port to start the Orchestrator on.
                Defaults to `6790`.
        """
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
                except Exception as f:
                    raise RuntimeError('Also failed to setup local environment!') from f
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
        rlxout.info("Use this command to shutdown database if not terminated correctly:")
        rlxout.info(f"$(smart dbcli) -h {db.hosts[0]} -p {port} shutdown", newline=False)

        return exp, db, entry_db

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
        if self.type == 'pbs':
            return os.environ['PBS_NODEFILE']
        if self.type == 'slurm':
            return os.environ['SLURM_NODELIST']
        raise NotImplementedError(
                f"Method get_hostlist not implemented for scheduler type {self.type}")

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
        """Whether `ClusterManager` runs in **Distributed** or **Local** mode.

        Returns:
            bool: `True` if in **Distributed Mode**, `False` otherwise.
        """
        return len(self._hosts) > 1

    @property
    def head(self) -> str:
        """Return name of Head node, which is where Relexi actually runs on.

        Returns:
            str: Hostname of the Head node.
        """
        return self.hosts[0]

    @property
    def workers(self) -> List[str]:
        """Returns list of Workers used for running training environments.

        Returns:
            list: List containing the hostnames of Worker nodes as strings.
        """
        if self.is_distributed:
            return self.hosts[1:]
        return self.hosts

    def info(self):
        """Print information about the current environment."""
        rlxout.info("Found the following environment:")
        rlxout.info(f"  Scheduler: {self.type}", newline=False)
        rlxout.info(f"  Hosts:     {self.hosts}", newline=False)
        if self.is_distributed:
            rlxout.info("Relexi is running in distributed mode:")
            rlxout.info(f"  Head:      {self.head}", newline=False)
            rlxout.info(f"  Workers:   {self.workers}", newline=False)
        else:
            rlxout.info(f"Relexi is running in local mode on: {self.head}")
