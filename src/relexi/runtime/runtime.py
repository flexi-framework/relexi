#!/usr/bin/env python3

"""The Runtime class for managing the HPC runtime environment."""

import os
import socket
import subprocess
from typing import List, Optional, Union

import smartsim
from smartsim import Experiment
from smartsim.database.orchestrator import Orchestrator

import relexi.io.output as rlxout
from relexi.runtime.helpers import generate_rankfile_ompi


class Runtime:
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
        db (Orchestrator): The launched `Orchestrator` database from the
            `smartsim` package.
        db_entry (str): IP address of the host of the database. Required to
            connect a client to the database.
        exp (Experiment): The `Experiment` object the `Orchestrator` is
            launched with.

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
            db_port: Optional[int] = 6790
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
        """
        # Using SmartSim utility to identify type automatically
        try:
            if type_ == 'auto':
                rlxout.info('Identifying environment...')
                scheduler = smartsim.wlm.detect_launcher()
                rlxout.info(f'Found "{scheduler}" environment!', newline=False)
                self.type = scheduler.casefold().strip()
            else:
                self.type = type_.casefold().strip()

            rlxout.info(f'Setting up "{self.type}" runtime...')
            self._hosts = self._get_hostlist()
        except Exception as e:
            rlxout.warning(f'Failed: {e}')
            if self.type != 'local':
                rlxout.info('Trying to setup LOCAL runtime instead...', newline=False)
                try:
                    self.type = 'local'
                    self._hosts = self._get_hostlist()
                except Exception as f:
                    raise RuntimeError('Also failed to setup LOCAL environment!') from f
            else:
                raise RuntimeError('Failed to setup LOCAL training environment!') from e
        rlxout.info('Success!', newline=False)

        self._db = None
        self._exp, self._db, self._db_entry = self._launch_orchestrator(
            port=db_port,
            network_interface=db_network_interface,
        )

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
            local_host = self._get_local_hostname()
            workers = self.hosts.copy()
            if local_host in workers:
                workers.remove(local_host)
            else:
                rlxout.warning(f'Localhost "{local_host}" not found in hosts list:')
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
        """Get the IP address of the host of the database.

        Returns:
            str: IP address of the host of the database.
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
        rlxout.small_banner('Starting Orchestrator...')

        # Generate flexi experiment
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

        entry_db = socket.gethostbyname(db.hosts[0])
        rlxout.info('Use this command to shutdown database if not terminated correctly:')
        rlxout.info(f'$(smart dbcli) -h {db.hosts[0]} -p {port} shutdown', newline=False)

        return exp, db, entry_db

    def _get_hostlist(self) -> List[str]:
        """Get the list of hosts the script is executed on.

        Uses the runtime type to determine the hostlist via the environment
        variables set by the corresponding scheduler environment.

        Returns:
            list: List containing the hostnames as strings.

        Raises:
            NotImplementedError: If the method is not implemented for the
                scheduler type.
        """
        if self.type == 'local':
            return [self._get_local_hostname()]
        if self.type == 'pbs':
            nodes = self._read_pbs_nodefile()
            # Get the list of unique nodes via casting into set and list again
            return list(set(nodes))
        if self.type == 'slurm':
            return self._get_slurm_nodelist()
        raise NotImplementedError(
            f'Method `get_hostlist` not implemented for runtime "{self.type}"!')

    def _read_pbs_nodefile(self) -> List[str]:
        """Read the `PBS_NODEFILE` and return the list of nodes.

        NOTE:
            The `PBS_NODEFILE` contains the list of nodes allocated to the job.
            If a node provides multiple MPI slots, it is the corresponding
            number of times in the file.

        Returns:
            list: List containing the hostnames as strings.
        """
        if self.type != 'pbs':
            raise ValueError('Method "read_pbs_nodefile" only available for PBS scheduler!')
        node_file = os.getenv('PBS_NODEFILE')
        if node_file is None:
            raise KeyError('Environment variable "PBS_NODEFILE" is not set!')
        with open(node_file, 'r', encoding='utf-8') as f:
            nodes = [line.strip() for line in f.readlines()]
        return nodes

    def _get_slurm_nodelist(self) -> List[str]:
        """Get the list of hosts from the SLURM_NODELIST environment variable.

        Returns:
            list: List containing the unique hostnames as strings.
        """
        if self.type != 'slurm':
            raise ValueError('Method "get_slurm_nodelist" only available for SLURM scheduler!')
        # Get the compressed list of nodes from SLURM_NODELIST
        node_list = os.getenv('SLURM_NODELIST')
        if node_list is None:
            raise KeyError('Environment variable "SLURM_NODELIST" is not set!')
        # Use scontrol to expand the node list
        result = subprocess.run(['scontrol', 'show', 'hostname', node_list], capture_output=True, text=True)
        # Check if the command was successful
        if result.returncode != 0:
            raise RuntimeError(f'scontrol command failed: {result.stderr.strip()}')
        # Split the output into individual hostnames
        return result.stdout.strip().split('\n')

    def _get_local_hostname(self) -> str:
        """Get the hostname of the machine executing the Python script.

        Returns:
            str: Hostname of the local machine executing the script.
        """
        return socket.gethostname()
