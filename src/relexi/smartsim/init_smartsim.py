#!/usr/bin/env python3

import os
import json
import socket
import subprocess

from smartsim import Experiment
from smartsim.database import Orchestrator

import relexi.io.output as rlxout


def get_host():
  """
  Get the host the script is executed on from the env variable
  """
  return  socket.gethostname()


def get_pbs_hosts():
  """
  Get the host list from the PBS Nodefile
  """
  nodefile_path = os.environ["PBS_NODEFILE"]

  with open(nodefile_path, "r") as f:
    hostlist = []
    for line in f:
      # only take the name not the entire ip-address otherwise there will be an error
      # it will set the command line flag "mpirun ... -host <hostname_here>"
      # This only works with the hostname shorthand
      full_host_ip = line.strip()             # e.g. abc.ib0...de
      hostname = full_host_ip.split(".")[0]   # e.g. abc
      if not (hostname in hostlist):
        hostlist.append(hostname)

  return hostlist


def get_pbs_walltime():
  """
  Get the walltime of the current PBS job
  """
  job_id = os.environ["PBS_JOBID"]
  cmd=f"qstat -xfF json {job_id}"
  stat_json_str = subprocess.check_output(cmd, shell=True, text=True)
  stat_json = json.loads(stat_json_str)

  return stat_json["Jobs"][job_id]["Resource_List"]["walltime"]



def init_smartsim(port = 6790
                 ,num_dbs = 1
                 ,NETWORK_INTERFACE="ib0"
                 ,launcher_type = "local"
                 ,orchestrator_type = "local"
                 ):
  """
  Initializes the smartsim architecture by starting the orchestrator, launching the experiment and get the list of hosts.

  NOTE 1:
    Combinations of Experiment launcher and orchestrator type:
    (TL;DR: Must be identical!)
    1. laun.: local, orch.: pbs   = incompatible
    2. laun.: local, orch.: local = only one in memory database possible, mpirun will still distribute the flexi instances to other nodes
    3. laun.: pbs,   orch.: pbs   = doesnt support clusters of size 2 otherwise works flawlessly (warning: orchestrator doesn't find the cluster configuration)
    4. laun.: pbs,   orch.: local = not supported error: not supported by PBSPro  
  """

  rlxout.printSmallBanner('Starting SmartSim...')

  # Check whether launcher and orchestrator are identical (case-insensitive)
  if not (launcher_type.casefold() == orchestrator_type.casefold()):
    rlxout.printWarning('Chosen Launcher '+launcher_type+' and orchestrator '+orchestrator_type +' are incompatible! Please choose identical types for both!')

  # Is database clustered, i.e. hosted on different nodes?
  db_is_clustered = num_dbs > 1

  # First try PBS if necessary. Use local configuration as backup
  if launcher_type.casefold() == 'pbs':
    PBS_failed = False
    try:
      # try to load the batch settings from the batch job environment variables like PBS_JOBID and PBS_NODEFILE
      walltime = get_pbs_walltime()
      hosts = get_pbs_hosts()
      num_hosts = len(hosts)
      rlxout.printNotice(f"Identified available nodes: {hosts}")

      # Maximum of 1 DB per node allowed for PBS Orchestrator
      if num_hosts < num_dbs:
        rlxout.printWarning(f"You selected {num_dbs} databases and {num_hosts} nodes, but maximum is 1 database per node. "+
                      "Setting number of databases to {num_hosts}")
        num_dbs = num_hosts

      # Clustered DB with PBS orchestrator requires at least 3 nodes for reasons
      if db_is_clustered:
        if num_dbs < 3:
          rlxout.printWarning(f"Only {num_dbs} databases requested, but clustered orchestrator requires 3 or more databases. "+
                        "Non-clustered orchestrator is launched instead!")
          db_is_clustered = False
        else:
          rlxout.printNotice(f"Using a clustered database with {num_dbs} instances.")
      else:
        rlxout.printNotice(f"Using an UNclustered database on root node.")

    except:
      # If there are no environment variables for a batchjob, then use the local launcher
      rlxout.printWarning(f"Didn't find pbs batch environment. Switching to local setup.")
      PBS_failed = True


  # If local configuration is required or if scheduler-based launcher failed.
  if (launcher_type.casefold() == 'local') or PBS_failed:
    launcher_type ="local"
    orchestrator_type = "local"
    db_is_clustered=False
    hosts = [get_host()]


  # Generate flexi experiment
  exp = Experiment("flexi", launcher=launcher_type)


  # Initialize the orchestrator based on the orchestrator_type
  if orchestrator_type.casefold() == "local":
    db = Orchestrator(
            port=port,
            interface='lo'
            )

  elif orchestrator_type.casefold() =="pbs":
    db = Orchestrator(
            launcher='pbs',
            port=port,
            db_nodes=num_dbs,
            batch=False, # false if it is launched in an interactive batch job 
            time=walltime, # this is necessary, otherwise the orchestrator wont run properly
            interface=NETWORK_INTERFACE,
            hosts=hosts, # this must be the hostnames of the nodes, it mustn't be the ip-addresses
            run_command="mpirun"
            )
  else:
    rlxout.printWarning("Orchester type "+orchestrator_type+" not implemented")


  # remove db files from previous run if necessary
  #if CLEAN_PREVIOUS_RUN:
  #  db.remove_stale_files()

  # startup Orchestrator
  rlxout.printNotice("Starting the Database...",newline=False)
  exp.start(db)

  # get the database nodes and select the first one
  entry_db = socket.gethostbyname(db.hosts[0])
  rlxout.printNotice(f"Identified 1 of {len(db.hosts)} database hosts to later connect clients to: {entry_db}",newline=False)
  rlxout.printNotice(f"If the SmartRedis database isn't stopping properly you can use this command to stop it from the command line:")
  for db_host in db.hosts:
    rlxout.printNotice(f"$(smart --dbcli) -h {db_host} -p {port} shutdown",newline=False)


  # If multiple nodes are available, the first executes ReLeXI, while 
  # all worker processes are started on different nodes.
  if len(hosts)>1:
    worker_nodes = hosts[1:]
  else: # Only single node
    worker_nodes = hosts


  return exp, worker_nodes, db, entry_db, db_is_clustered
