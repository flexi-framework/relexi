#!/usr/bin/env python3

"""The ClusterManager class for managing the HPC environment."""

import os
import socket
import subprocess
from typing import List, Optional

import smartsim
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

    TYPES = ['local', 'pbs', 'slurm']  # All implemented types of cluster managers

    def __init__(
            self,
            scheduler_type: Optional[str] = 'auto',
            db_network_interface: Optional[str] = 'lo',
            db_port: Optional[int] = 6790
            ):
        """Initialize the ClusterManager.

        Args:
            scheduler_type (str, optional): Type of the cluster manager.
                Must be `'local'`, `'pbs'`, `'slurm'` or `'auto'`. Defaults to
                `'auto'`, for which the type of cluster environment iss
                identified automatically.
            db_network_interface (str, optional): Network interface to use for
                the Orchestrator. Defaults to `'lo'`.
            db_port (int, optional): Port to start the Orchestrator on.
                Defaults to `6790`.
        """
        # Using SmartSim utility to identify type automatically
        if scheduler_type == 'auto':
            rlxout.info('Trying to identify cluster environment...')
            scheduler = smartsim.wlm.detect_launcher()
            rlxout.info(f'Found "{scheduler}" environment!', newline=False)
            self.type = scheduler.casefold().strip()
        else:
            self.type = scheduler_type.casefold().strip()

        rlxout.info(f'Trying to setup "{self.type}" environment...')
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
        """Prints information about the current environment."""
        rlxout.info("Found the following environment:")
        rlxout.info(f"  Scheduler: {self.type}", newline=False)
        rlxout.info(f"  Hosts:     {self.hosts}", newline=False)
        if self.is_distributed:
            rlxout.info("Relexi is running in distributed mode:")
            rlxout.info(f"  Head:      {self.head}", newline=False)
            rlxout.info(f"  Workers:   {self.workers}", newline=False)
        else:
            rlxout.info(f"Relexi is running in local mode on: {self.head}")

    def get_worker_slots(self) -> List[str]:
        """Gets the list of available MPI slots on the Worker nodes.

        To obtain a list of all available MPI slots on the Worker nodes, the
        following strategy is used depending on the type of environment:
            - `local`: All CPU cores from localhost are used except one to run
                training script and the database.
            - `pbs`: The number of slots is determined by accessing the
                `PBS_NODEFILE` environment variable and removing the Head node.
            - `slurm`: The number of slots is determined by accessing the
                `SLURM_JOB_CPUS_PER_NODE` environment variable and counting the
                number of Worker nodes.

        Returns:
            list: List containing hostname and host-local slot number of each
                free slot on the Worker nodes.
        """
        if self.type == 'local':
            n_cpus = os.cpu_count()-1  # Save 1 CPU core for the Head tasks
            return [[self.head, str(i)] for i in range(n_cpus)]

        if self.type == 'pbs':
            # Get PBS_NODEFILE count number of slots per node and return list
            # of slots per node.
            nodes = self._read_pbs_nodefile()
            worker_slots = []
            for worker in self.workers:
                n_slots = sum(1 for nodename in nodes if worker in nodename)
                #worker_slots.append({worker: str(i)})  # Dict of slots per node
                for i in range(n_slots):
                    worker_slots.append([worker, str(i)])
            return worker_slots

        if self.type == 'slurm':
            cpus_per_node = os.environ['SLURM_JOB_CPUS_PER_NODE']
            if cpus_per_node is None:
                raise KeyError("Environment variable 'SLURM_JOB_CPUS_PER_NODE' is not set!")
            for worker in self.workers:
                for i in range(cpus_per_node):
                    worker_slots.append([worker, str(i)])
            return worker_slots

        raise NotImplementedError(
            f"Method 'get_worker_slots' not implemented for scheduler type {self.type}")

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

        Checks for the number of hosts available. If more than one host is
        used, the `ClusterManager` runs in **Distributed Mode**.

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
        return self._get_local_hostname()

    @property
    def workers(self) -> List[str]:
        """Returns list of Workers used for running training environments.

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
                rlxout.warning(f"Localhost '{local_host}' not found in hosts list:")
                rlxout.warning(f"  {workers}")
            return workers
        return self.hosts

    @property
    def db(self) -> Orchestrator:
        """Get the Orchestrator database object.

        Returns:
            Orchestrator: The Orchestrator database object.
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
        """Get the Experiment object the Orchestrator is launched with.

        Returns:
            Experiment: The Experiment object.
        """
        return self._exp

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
            return [self._get_local_hostname()]
        if self.type == 'pbs':
            nodes = self._read_pbs_nodefile()
            # Get the list of unique nodes via casting into set and list again
            return list(set(nodes))
        if self.type == 'slurm':
            return self._get_slurm_nodelist()
        raise NotImplementedError(
            f"Method `get_hostlist` not implemented for scheduler type {self.type}!")

    def _read_pbs_nodefile(self) -> List[str]:
        """Read the PBS_NODEFILE and return the list of nodes.

        NOTE:
            The PBS_NODEFILE contains the list of nodes allocated to the job.
            If a node provides multiple MPI slots, it is the corresponding
            number of times in the file.

        Returns:
            list: List containing the hostnames as strings.
        """
        if self.type != 'pbs':
            raise ValueError("Method 'read_pbs_nodefile' only available for PBS scheduler!")
        node_file = os.environ['PBS_NODEFILE']
        if node_file is None:
            raise KeyError("Environment variable 'PBS_NODEFILE' is not set!")
        with open(node_file, 'r', encoding='utf-8') as f:
            nodes = [line.strip() for line in f.readlines()]
        return nodes

    def _get_slurm_nodelist(self) -> List[str]:
        """Get the list of hosts from the SLURM_NODELIST environment variable.

        Returns:
            list: List containing the unique hostnames as strings.
        """
        if self.type != 'slurm':
            raise ValueError("Method 'get_slurm_nodelist' only available for SLURM scheduler!")
        # Get the compressed list of nodes from SLURM_NODELIST
        node_list = os.getenv('SLURM_NODELIST')
        if node_list is None:
            raise KeyError("Environment variable 'SLURM_NODELIST' is not set!")
        # Use scontrol to expand the node list
        result = subprocess.run(['scontrol', 'show', 'hostname', node_list], capture_output=True, text=True)
        # Check if the command was successful
        if result.returncode != 0:
            raise RuntimeError(f"scontrol command failed: {result.stderr.strip()}")
        # Split the output into individual hostnames
        return result.stdout.strip().split('\n')

    def _get_local_hostname(self) -> str:
        """Get the hostname of the machine executing the Python script.

        Returns:
            str: Hostname of the local machine executing the script.
        """
        return socket.gethostname()
