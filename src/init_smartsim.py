#!/usr/bin/env python3

"""
Module to initialise the SmartSim Experiment and the SmartSim Orchestrator.

Experiments are the Python user interface for SmartSim.
Experiment is a factory class that creates stages of a workflow
and manages their execution.
The instances created by an Experiment represent executable code
that is either user-specified, like the ``Model`` instance created
by ``Experiment.create_model``, or pre-configured, like the ``Orchestrator``
instance created by ``Experiment.create_database``.
Experiment methods that accept a variable list of arguments, such as
``Experiment.start`` or ``Experiment.stop``, accept any number of the
instances created by the Experiment.
In general, the Experiment class is designed to be initialized once
and utilized throughout runtime.

The Orchestrator is an in-memory database that can be launched
alongside entities in SmartSim. Data can be transferred between
entities by using one of the Python, C, C++ or Fortran clients
within an entity.
"""

import os
import json
import socket
import subprocess

from smartsim import Experiment
from smartsim.database import Orchestrator, SlurmOrchestrator

from output import printWarning, printNotice
from output import printHeader, printBanner, printSmallBanner


def get_host():
    """
    Get the host the script is executed on from the env variable
    """
    return socket.gethostname()


def get_slurm_hosts():
    """
    Get the host list from the SLURM_JOB_NODELIST environment variable
    """
    hostslist_str = subprocess.check_output(
        "scontrol show hostnames", shell=True, text=True
    )
    return list(set(hostslist_str.split("\n")[:-1]))  # returns unique name of hosts


def get_slurm_walltime():
    """
    Get the walltime of the current SLURM job
    """
    cmd = 'squeue -h -j $SLURM_JOBID -o "%L"'
    return subprocess.check_output(cmd, shell=True, text=True)


def init_smartsim(
    port=6790,
    num_dbs=1,
    network_interface="ib0",
    launcher_type="local",
    orchestrator_type="local",
    run_command="mpirun",
):
    """
    Initializes the smartsim architecture by starting the orchestrator, launching the experiment, and get the list of hosts.

    NOTE 1:
      Combinations of Experiment launcher and Orchestrator type:
      (TL;DR: Must be identical!)
      1. laun.: local, orch.: slurm = incompatible
      2. laun.: local, orch.: local = only one in memory database possible, mpirun will still distribute the flexi instances to other nodes
      3. laun.: slurm, orch.: slurm = doesnt support clusters of size 2 otherwise works flawlessly (warning: orchestrator doesn't find the cluster configuration)
      4. laun.: slurm, orch.: local = not supported error: not supported by PBSPro
    """

    print("Starting SmartSim...")

    # Check whether launcher and orchestrator are identical (case-insensitive)
    if not launcher_type.casefold() == orchestrator_type.casefold():
        raise ValueError(
            f"Chosen Launcher {launcher_type} and orchestrator {orchestrator_type} are incompatible! \
            Please choose identical types for both!"
        )

    # Is database clustered, i.e. hosted on different nodes?
    db_is_clustered = num_dbs > 1

    # First try Slurm if necessary. Use local configuration as backup
    if launcher_type.casefold() == "slurm":
        slurm_failed = False
        try:
            # Get slurm settings
            hosts = get_slurm_hosts()
            walltime = get_slurm_walltime()
            num_hosts = len(hosts)
            print(f"Identified available nodes: {hosts}")

            # Maximum of 1 DB per node allowed for Slurm Orchestrator
            if num_hosts < num_dbs:
                print(
                    f"You selected {num_dbs} databases and {num_hosts} nodes, but maximum is 1 database per node.\
                    \nSetting number of databases to {num_hosts}"
                )
                num_dbs = num_hosts

            # Clustered DB with Slurm orchestrator requires at least 3 nodes for reasons
            if db_is_clustered:
                if num_dbs < 3:
                    print(
                        f"Only {num_dbs} databases requested, but clustered orchestrator requires 3 or more databases.\
                \nNon-clustered orchestrator is launched instead!"
                    )
                    db_is_clustered = False
                else:
                    print(f"Using a clustered database with {num_dbs} instances.")
            else:
                print("Using an UNclustered database on root node.")

        except:
            # If there are no environment variables for a batchjob, then use the local launcher
            print("Didn't find SLURM batch environment. Switching to local setup.")
            slurm_failed = True

    # If local configuration is required or if scheduler-based launcher failed.
    if (launcher_type.casefold() == "local") or slurm_failed:
        launcher_type = "local"
        orchestrator_type = "local"
        db_is_clustered = False
        hosts = [get_host()]

    # Generate Sod2D experiment
    exp = Experiment("sod2d", launcher=launcher_type)

    # Initialize the orchestrator based on the orchestrator_type
    if orchestrator_type.casefold() == "local":
        db = Orchestrator(port=port, interface="lo")
    elif orchestrator_type.casefold() == "slurm":
        db = SlurmOrchestrator(
            port=port,
            db_nodes=num_dbs, # SlurmOrchestrator supports multiple databases per node
            batch=False,  # db_is_clustered, # false if it is launched in an interactive batch job
            time=walltime,  # this is necessary, otherwise the orchestrator wont run properly
            interface=network_interface,
            hosts=hosts,  # specify hostnames of nodes to launch on (it mustn't be the ip-addresses)
            run_command=run_command,  # specify launch binary. Options are "mpirun" and "srun", defaults to "srun"
            db_per_host=1,  # number of database shards per system host (MPMD), defaults to 1
            single_cmd=True,  # run all shards with one (MPMD) command, defaults to True
        )
    else:
        raise ValueError(f"Orchester type {orchestrator_type} not implemented.")

    # remove db files from previous run if necessary
    # if CLEAN_PREVIOUS_RUN:
    #  db.remove_stale_files()

    # startup Orchestrator
    print("Starting the Database...")
    exp.start(db)

    # # get the database nodes and select the first one
    # entry_db = socket.gethostbyname(db.hosts[0])
    # print(f"Identified 1 of {len(db.hosts)} database hosts to later connect clients to: {entry_db}")

    print("If the SmartRedis database isn't stopping properly you can use this command \
    to stop it from the command line:")
    for db_host in db.hosts:
        print(f"$(smart --dbcli) -h {db_host} -p {port} shutdown")

    # # If multiple nodes are available, the first executes ReLeXI, while
    # # all worker processes are started on different nodes.
    # if len(hosts)>1:
    #   worker_nodes = hosts[1:]
    # else: # Only single node
    #   worker_nodes = hosts
    # return exp, worker_nodes, db, entry_db, db_is_clustered

    # we run the model on a single host, and have
    return exp, hosts, db, db_is_clustered
