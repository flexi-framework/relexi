#!/usr/bin/env python3

"""The Cluster Manager contains all information about the HPC environment."""

import socket
import rlxout
import smartsim

import sys

class ClusterManager():
    """Base class for cluster managers.

    This class defines the interface for cluster managers. which contain all
    information of the HPC environment used for the training environments. In
    particular, it provides properties to access the hostnames and entails the
    launched Orchestrator.

    Properties:
        type (str): Type of the cluster manager.
        hosts (list): List of hostnames.
        dedicated_head (bool): Whether dedicated head and worker nodes are
            available or everything runs on single shared node.
        head (str): Hostname of Head node.
        workers (list): List of worker nodes.
        job_id (str): Job ID of the current job.

    Methods:
        print_info: Print information about the current environment.
        generate_rankfiles: Generate rank files for OpenMPI process binding.

    Raises:
        NotImplementedError: If the methods are not implemented for the
            provided scheduler type.
        ValueError: If the scheduler type is not supported.
    """

    TYPES = ['local', 'pbs', 'slurm']  # All possible types of cluster managers

    def __init__(
            self,
            scheduler_type='local',
            db_port=6790,
            db_network_interface='lo',
            ):

        # Check if the scheduler type is supported
        self.type = scheduler_type.casefold().strip()

        rlxout.info('Trying to identify {self.type} training environment...')
        try:
            self._hosts = self._get_hostlist()
        except:
            if self.type == 'local':
                raise RuntimeError('Failed to setup local training environment!')
            rlxout.error('Failed!')
            rlxout.info('Trying to run in local mode instead...')
            try:
                self.type = 'local'
                self._hosts = self._get_hostlist()
            except:
                raise RuntimeError('Also failed to setup local environment!')
        rlxout.info('Success!')

        self.exp, self.db = self._launch_orchestrator(port=db_port,
                                                      network_interface=db_network_interface)

    def _launch_orchestrator(self, port, network_interface):
        """Launches the SmartSim Orchestrator for the current job.

        Args:
            port (int): Port to start the Orchestrator on.
            network_interface (str): Network interface to use for the Orchestrator.

        Returns:
            Experiment: The Experiment object.
            Orchestrator: The Orchestrator object.
        """
        rlxout.small_banner('Starting Orchestrator...')

        # Generate flexi experiment
        exp = smartsim.Experiment('flexi', launcher=launcher_type)

        # Initialize the orchestrator based on the orchestrator_type
        if self.type == 'local':
            db = smartsim.Orchestrator(launcher=self.type, port=port, interface='lo')
        elif self.type in {'pbs','slurm'}:
            db = smartsim.Orchestrator(launcher=self.type, port=port, interface=network_interface)
        else:
            raise NotImplementedError(f"Orchestrator type {self.type} not implemented.")

        # startup Orchestrator
        rlxout.info("Starting the Database...", newline=False)
        try:
            exp.start(db)
        except Exception as e:
            raise RuntimeError(f"Failed to start the Orchestrator: {e}")
        rlxout.info("  Success!")

        # get the database nodes and select the first one
        entry_db = socket.gethostbyname(db.hosts[0])
        rlxout.info("If the SmartRedis database isn't stopping properly you can use this command to stop it from the command line:")
        rlxout.info(f"$(smart dbcli) -h {db.hosts[0]} -p {port} shutdown", newline=False)

        return exp, db

    def print_info(self):
        """Print information about the current job."""
        rlxout.info("Found the following environment information:")
        rlxout.info(f"Scheduler Type: {self.type}")
        rlxout.info(f"Job ID: {self.job_id}")
        rlxout.info(f"Hosts: {self.hosts}")
        rlxout.info(f"Head node: {self.head}")
        rlxout.info(f"Worker nodes: {self.workers}")

    def _get_hostlist(self):
        """Get the list of hosts the script is executed on.

        Uses the scheduler type to determine the hostlist via the environment
        variables set by the scheduler.

        Returns:
            List containing the hostnames as strings.

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
    def type(self):
        """Get the type of scheduler environment used for the cluster manager.

        Returns:
            Type of the cluster manager as a string.
        """
        return self._type

    @type.setter
    def type(self, value):
        """Set the type of scheduler environment used for the cluster manager.
        Ensure that the type is supported.

        Args:
            value (str): Type of the cluster manager as a string.
        """
        if not value in self.TYPES:
            raise ValueError(f"Scheduler type {value} not supported.")
        self._type = value

    @property
    def hosts(self):
        """Get the list of hosts the script is executed on.

        Returns:
            List containing the hostnames as strings.
        """
        return self._hosts

    @property
    def dedicated_head(self):
        """Return whether dedicated head and worker nodes are available or
        everything runs on single shared node.

        Returns:
            Bool indicating whether dedicated head is used or not.
        """
        return len(self._hosts) > 1

    @property
    def head(self):
        """Get head node, which is first node, i.e. node Relexi is actually
        started on.

        Returns:
            Hostname of Head node as string.
        """
        return self._hosts[0]

    @property
    def workers(self):
        """Get a list of worker nodes.

        Returns:
            List containing the hostnames of worker nodes as strings.
        """
        if self.dedicated_head:
            return self._hosts[1:]
        return self._hosts

    @property
    def job_id(self):
        """Get the jobID of the current job.

        Returns:
            Job ID of the current job.
        """
        return self._job_id

    def generate_rankfile(self, n_par_env, ranks_per_env, base_path=None):
        """Generate rank file for OpenMPI process binding.

        Args:
            cores_per_node (int): Number of cores per node.
            n_par_env (int): Number of parallel environments to be launched.
            ranks_per_env (int): Number of ranks per environments.
            base_path (str): (Optional.) Path to the directory of the rank files.

        Returns:
            List of filenames of the rankfiles.
        """
        if self.type == 'local':
            return generate_rankfile_ompi(self.hosts, cores_per_node, n_par_env, ranks_per_env, base_path)
        if self.type == 'pbs':
            return generate_rankfile_ompi(self.hosts, cores_per_node, n_par_env, ranks_per_env, base_path)
        if self.type == 'slurm':
            return generate_rankfile_ompi(self.hosts, cores_per_node, n_par_env, ranks_per_env, base_path)
        else:
            raise NotImplementedError(f"Method generate_rankfile not implemented for scheduler type {self.type}")

if __name__ == '__main__':
    db = ClusterManager()
