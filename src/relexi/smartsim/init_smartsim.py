#!/usr/bin/env python3

"""Helpers for launching the SmartSim Orchestrator."""

import os
import json
import socket
import subprocess

from smartsim import Experiment
from smartsim.database import Orchestrator

import relexi.io.output as rlxout


def get_host():
    """Get the host the script is executed on from the env variable.

    Returns:
        Hostname as string
    """
    return socket.gethostname()


def get_pbs_hosts():
    """Get the host list from the PBS Nodefile.

    Returns:
        List containing the hostnames as strings
    """
    nodefile_path = os.environ["PBS_NODEFILE"]
    with open(nodefile_path, "r", encoding='ascii') as f:
        hostlist = []
        for line in f:
            # only take the name not the entire ip-address otherwise there will be an error
            # it will set the command line flag "mpirun ... -host <hostname_here>"
            # This only works with the hostname shorthand
            full_host_ip = line.strip()             # e.g. abc.ib0...de
            hostname = full_host_ip.split(".")[0]   # e.g. abc
            if not hostname in hostlist:
                hostlist.append(hostname)
    return hostlist


def get_pbs_walltime():
    """Get the walltime of the current PBS job.

    Returns:
       Walltime of current PBS job.
    """
    job_id = os.environ["PBS_JOBID"]
    cmd = f"qstat -xfF json {job_id}"
    stat_json_str = subprocess.check_output(cmd, shell=True, text=True)
    stat_json = json.loads(stat_json_str)
    return stat_json["Jobs"][job_id]["Resource_List"]["walltime"]


def init_smartsim(
    port=6790,
    num_dbs=1,
    network_interface="ib0",
    launcher_type="local",
    orchestrator_type="local"
):
    """Starts the orchestrator, launches an experiment and gets list of hosts.

    Args:
        port (int): (Optional.) Port number on which Orchestrator will be
            launched.
        num_dbs (int): (Optional.) Number of databases should be launched.
            `num_dbs>1` imply that the database is clustered , i.e. distributed
            across multiple instances.
        network_interface (string) = (Optional.) Name of network interface to
            be used to establish communication to clients.
        launcher_type (string): (Optional.) Launcher to be used to start the
            executable. Currently implemented are:
                * local
                * mpirun
        orchestrator_type (string): Scheduler environment in which the
            orchestrator is launched. Currently implemented are:
                * local
                * pbs
    Returns:
        smartsim.Experiment: The experiments in which the Orchestrator was
            started
        list: List of names of the nodes used as workers to run the simulations
        smarsim.Orchestrator: The launched Orchestrator
        string: The IP address and port used to access the Orchestrator
        bool: Flag to indicate whether Orchestrator is clustered.

    Note:
        Admissable combinations of Experiment launcher and orchestrator type:
            * laun.: local, orch.: pbs = incompatible.
            * laun.: local, orch.: local = only 1 in-memory database possible.
                `mpirun` will still distribute the flexi instances to other
                nodes.
            * laun.: pbs, orch.: pbs = does not support clusters of size 2
                otherwise works flawlessly (warning: orchestrator doesn't find
                the cluster configuration).
            * laun.: pbs, orch.: local = not supported error: not supported by
                PBSPro.

    TODO:
        * Add support for SLURM.
        * Clean implementation and nesting.
        * Make object out of this.
        * Allow to reconnect to already started Orchestrator
        * Or closue Orchestrator still open from previous run
    """

    rlxout.small_banner('Starting SmartSim...')

    # Check whether launcher and orchestrator are identical (case-insensitive)
    if not launcher_type.casefold() == orchestrator_type.casefold():
        rlxout.warning(f'Chosen Launcher {launcher_type} and orchestrator {orchestrator_type} are incompatible! Please choose identical types for both!')

    # Is database clustered, i.e. hosted on different nodes?
    db_is_clustered = num_dbs > 1

    # First try PBS if necessary. Use local configuration as backup
    pbs_failed = False
    if launcher_type.casefold() == 'pbs':
        try:
            # try to load the batch settings from the batch job environment
            # variables like PBS_JOBID and PBS_NODEFILE
            walltime = get_pbs_walltime()
            hosts = get_pbs_hosts()
            num_hosts = len(hosts)
            rlxout.info(f"Identified available nodes: {hosts}")

            # Maximum of 1 DB per node allowed for PBS Orchestrator
            if num_hosts < num_dbs:
                rlxout.warning(f"You selected {num_dbs} databases and {num_hosts} nodes, but maximum is 1 database per node. Setting number of databases to {num_hosts}")
                num_dbs = num_hosts

            # Clustered DB with PBS orchestrator requires at least 3 nodes for reasons
            if db_is_clustered:
                if num_dbs < 3:
                    rlxout.warning(f"Only {num_dbs} databases requested, but clustered orchestrator requires 3 or more databases. Non-clustered orchestrator is launched instead!")
                    db_is_clustered = False
                else:
                    rlxout.info(f"Using a clustered database with {num_dbs} instances.")
            else:
                rlxout.info("Using an UNclustered database on root node.")

        except Exception:
            # If no env. variables for batchjob, use the local launcher
            rlxout.warning("Didn't find pbs batch environment. Switching to local setup.")
            pbs_failed = True

    # If local configuration is required or if scheduler-based launcher failed.
    if (launcher_type.casefold() == 'local') or pbs_failed:
        launcher_type = "local"
        orchestrator_type = "local"
        db_is_clustered = False
        hosts = [get_host()]

    # Generate flexi experiment
    exp = Experiment("flexi", launcher=launcher_type)

    # Initialize the orchestrator based on the orchestrator_type
    if orchestrator_type.casefold() == "local":
        db = Orchestrator(
            port=port,
            interface='lo'
        )

    elif orchestrator_type.casefold() == "pbs":
        db = Orchestrator(
            launcher='pbs',
            port=port,
            db_nodes=num_dbs,
            batch=False,  # false if it is launched in an interactive batch job
            time=walltime,  # this is necessary, otherwise the orchestrator wont run properly
            interface=network_interface,
            hosts=hosts,  # this must be the hostnames of the nodes, it mustn't be the ip-addresses
            run_command="mpirun"
        )
    else:
        rlxout.warning(f"Orchester type {orchestrator_type} not implemented!")
        raise NotImplementedError

    # startup Orchestrator
    rlxout.info("Starting the Database...", newline=False)
    exp.start(db)

    # get the database nodes and select the first one
    entry_db = socket.gethostbyname(db.hosts[0])
    rlxout.info(f"Identified 1 of {len(db.hosts)} database hosts to later connect clients to: {entry_db}", newline=False)
    rlxout.info("If the SmartRedis database isn't stopping properly you can use this command to stop it from the command line:")
    for db_host in db.hosts:
        rlxout.info(f"$(smart dbcli) -h {db_host} -p {port} shutdown", newline=False)

    # If multiple nodes are available, the first executes Relexi, while
    # all worker processes are started on different nodes.
    if len(hosts) > 1:
        worker_nodes = hosts[1:]
    else:  # Only single node
        worker_nodes = hosts

    return exp, worker_nodes, db, entry_db, db_is_clustered
