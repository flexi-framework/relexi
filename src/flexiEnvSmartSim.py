#!/usr/bin/env python3

import os
import csv
import time
import random
import numpy as np

from output import printWarning

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from smartsim import Experiment
from smartsim.database import PBSOrchestrator,Orchestrator
from smartsim.settings import MpirunSettings,RunSettings

from smartredis import Client

# Try to load matplotlib
try:
  import matplotlib.pyplot as plt
  MATPLOTLIB_FOUND = True
except ImportError:
  MATPLOTLIB_FOUND = False
  printWarning('Could not import Matplotlib. No figures will be created!')

if MATPLOTLIB_FOUND:
  import io
  import tensorflow as tf



class flexiEnv(py_environment.PyEnvironment):
  """
  This FLEXI environment is implemented as standard python environment,
  which should be compatible with most Reinforcement Learning libraries.
  It leverages the shared FLEXI library to perform simulation steps and
  call the init and finalize routines.
  """

  """ Initialize variables deleted in the destructor to ensure clean exit of ReLeXI """
  #db    = None
  #exp   = None
  #flexi = None
  tag=None

  def __init__( self
               ,exp
               ,flexi_path
               ,prm_file
               ,spectra_file
               ,reward_kmin
               ,reward_kmax
               ,reward_scale
               ,n_procs = 1
               ,n_envs = 1
               ,n_procs_per_node = 1
               ,restart_files=None
               ,random_restart_file=True
               ,entry_db = "127.0.0.1"
               ,port = 6780
               ,is_db_cluster = False
               ,debug=0
               ,tag = None
               ,hosts = None
               ,rankfiles = None
               ,mpi_launch_mpmd = False
               ,env_launcher = 'mpirun'
               ):
    """Initialize TF and FLEXI specific properties"""

    # Save values for reward function
    self.reward_kmin = reward_kmin
    self.reward_kmax = reward_kmax
    self.reward_scale = reward_scale

    # Path to FLEXI executable
    self.n_envs = n_envs
    self.n_procs = n_procs
    self.n_procs_per_node = n_procs_per_node
    self.prm_file = prm_file
    self.flexi_path = flexi_path
    self.hosts = hosts
    self.rankfiles = rankfiles

    # Sanity Check Launcher
    self.env_launcher = env_launcher
    if ((self.env_launcher == 'local') and (n_procs != 1)):
      printWarning("For env_launcher 'local', only single execution is allowed! Setting 'n_procs=1'!")
      printWarning("To run evironments in parallel with MPI, use env_launcher='mpi'!")
      n_procs = 1

    if (self.env_launcher == 'mpirun'):
      self.mpi_launch_mpmd = mpi_launch_mpmd
    else:   
      self.mpi_launch_mpmd = False

    # Save list of restart files
    self.random_restart_file = random_restart_file
    self.restart_files = restart_files

    # Read target DNS spectra from file
    if spectra_file:
      with open(spectra_file,'r',encoding='ascii') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        col_E = next(reader).index('E')
        E = []
        for rows in reader:
          E.append(float(rows[col_E]))
      self.E_DNS = E


    # Get experiment handle and port of db
    self.exp  = exp
    self.port = port
    self.entry_db = entry_db # This should be an ip-address not the hostname, because "-.,'" in the hostname will cause a crash
    self.is_db_cluster = is_db_cluster


    # Connect python redis client to an orchestrator database
    self.client = Client(address=f"{self.entry_db}:{str(self.port)}", cluster=self.is_db_cluster)

    # Build tag from tag plus env number
    if tag:
      self.tag  = [tag+str(i)+'_' for i in range(self.n_envs)]

    # Create Ompen-MPI rankfile
    #base_path = "/lustre/hpe/ws10/ws10.0/ws/hpchppof-relexi/testcase_philipp" 
    #print("init, n_envs: ", self.n_envs)
    #self.rankfiles = self._generate_rankefile_hawk_ompi(self.hosts, self.n_procs_per_node, self.n_envs, n_procs, base_path)

    # Startup FLEXI instances inside experiment to get state size
    self.flexi = self._start_flexi(self.exp,self.n_procs,self.n_envs)

    # Get current state from FLEXI environment
    self._state = self._get_current_state()

    # End FLEXI again. Otherwise it will compute the entire run...
    self._end_flexi()

    # Specify action and observation dimensions (neglect first batch dimension)
    self._action_spec = array_spec.ArraySpec(
        shape=(self._state.shape[1],),
        dtype=np.float32, name='action')
    self._observation_spec = array_spec.ArraySpec(
        shape=self._state.shape[1:],
        dtype=np.float32, name='observation')
    self._episode_ended = False


  def __del__(self):
    """Properly finalize all launched FLEXI instances within the SmartSim experiment."""
    self.stop()


  @property
  def batched(self):
    """ Override batched property to indicate that this environment is batched """
    return True


  @property
  def batch_size(self):
    """ Override batch size property according to chosen batch size """
    return self.n_envs


  def stop(self):
    """Stops all flexi instances inside launched in this environment."""
    if self.exp:
      self._stop_flexi_instances(self.exp)


  def start(self):
    """Starts all flexi instances with configuration specified in initialization."""
    # Start new FLEXI instance and get initial state
    self.flexi = self._start_flexi(self.exp,self.n_procs,self.n_envs)
    self._state = self._get_current_state()


  def _stop_flexi_instances(self,exp):
    """Remove experiment with SmartSim by finishing the FLEXI instances and stopping the Redis database."""
    if (self.flexi):
      for flexi in self.flexi:
        if not exp.finished(flexi):
          exp.stop(flexi)


  def _start_flexi(self,exp,n_procs,n_envs):
    """Start FLEXI instances within SmartSim experiment"""

    # Build hostlist to specify on which hosts each flexi is started
    # First check: Are there enough free ranks?
    ranks_avail  = self.n_procs_per_node*len(self.hosts)
    ranks_needed = n_envs*n_procs
    if (ranks_needed > ranks_avail):
      printWarning(f'Only {ranks_avail} ranks are available, but {ranks_needed} would be required '+
                    'to start {n_envs} on {n_procs} each.')

    # Distribute ranks to instances in a round robin fashion
    # TODO: Get ranks directly from hostfile for PBS Orchestrator
    hosts_per_flexi = np.zeros((n_envs,2),dtype=np.int8)
    n_cores_used  = 0
    for i in range(n_envs):
      # 1. Get first node the instance has ranks on
      hosts_per_flexi[i,0] = n_cores_used // self.n_procs_per_node
      # 2. Increase amount of used cores accordingly
      n_cores_used = n_cores_used + n_procs
      # 3. Get last node the instance has ranks on
      hosts_per_flexi[i,1] = (n_cores_used-1) // self.n_procs_per_node    # last  node


    flexi = []
    # Build list of individual FLEXI instances
    for i in range(n_envs):

      # Select (possibly random drawn) restart file for the run
      if self.random_restart_file:
        restart_file = random.choice(self.restart_files)
      else:
        restart_file = self.restart_files[0]

      args = [self.prm_file,restart_file]
      if self.tag[i]:
        # Tags are given to FLEXI with the Syntax "--tag [value]"
        args.append('--tag')
        args.append(self.tag[i])

      if (self.env_launcher == 'mpirun'):
        run_args = {"rankfile" : self.rankfiles[i] ,"report-bindings" : ""}
        run = MpirunSettings(exe=self.flexi_path, exe_args=args, run_args=run_args)
        run.set_tasks(n_procs)
        run.set_hostlist(self.hosts[hosts_per_flexi[i,0]:hosts_per_flexi[i,1]+1])

        # Create MPMD Settings and start later in single command
        if self.mpi_launch_mpmd:
          if i==0:
            f_mpmd = run
          else:
            f_mpmd.make_mpmd(run)

      else: # Otherwise do not use launcher
        run = RunSettings(exe=self.flexi_path, exe_args=args)

      # Create and directly start FLEXI instances
      if not self.mpi_launch_mpmd:
        flexi_instance = exp.create_model(self.tag[i]+"flexi", run)
        exp.start(flexi_instance, block=False,summary=False)
        flexi.append(flexi_instance)


    # Create MPMD Model from settings and start
    if self.mpi_launch_mpmd:
      flexi = exp.create_model(self.tag[0]+"flexi", f_mpmd)
      exp.start(flexi, block=False,summary=False)
      flexi = [flexi]

    return flexi


  def _end_flexi(self):
    """Stop FLEXI experiment with SmartSim"""
    for flexi in self.flexi:
      if not self.exp.finished(flexi):
        self.exp.stop(flexi)


  def _reset(self):
    """
    Resets the FLEXI environment by finalizing it and initializing it again.
    TODO: This is now done outside of the reset function by calling manually
          the functions "env.start()" and "env.stop"
    """

    ## Close FLEXI instance
    #self._end_flexi()
    self._episode_ended = False

    ## Start new FLEXI instance and get initial state
    #self.flexi = self._start_flexi(self.exp,self.n_procs,self.n_envs)
    #self._state = self._get_current_state()

    return ts.restart(self._state,batch_size=self.n_envs)


  def _step(self, action):
    """
    Performs a single step in the FLEXI environment. To this end, the actions of the agents
    based on the current flow state are set as Smagorinsky parameter Cs in FLEXI. We then
    perform some timestepping, get the reward of the new flow state, update the
    current flow state and return the transition.
    ."""
    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start a new episode.
      return self.reset()

    # Update Prediction
    self._set_prediction(action)

    # Poll New State
    # ATTENTION: HERE THE FLEXI TIMESTEPPING OCCURS!
    self._state = self._get_current_state()

    # Get Reward
    reward = self._get_reward()

    # Determine if simulation finished
    self._episode_ended = self._flexi_ended()

    # Return transition
    if self._episode_ended:
      return ts.termination(self._state, reward)
    else:
      # Discount is later multiplied with global discount from user input
      return ts.transition(self._state, reward, discount=np.ones((self.n_envs,)))


  def observation_spec(self):
    return self._observation_spec


  def action_spec(self):
    return self._action_spec


  def _set_prediction(self,action):
    """ Write action for current environment state in to the Database to be polled by the FLEXI client."""
    for i in range(self.n_envs):
      dataset = self.client.put_tensor(self.tag[i]+"Cs",action[i,::].astype(np.float64))


  def _get_current_state(self):
    """
    Get current flow state from the database.
    ATTENTION: This is the routine the enviroment will idle until the necessary data becomes available 
    """
    do_init = True
    for tag in self.tag:
      self.client.poll_tensor(tag+"U",10,10000)
      try:
        data = self.client.get_tensor(tag+"U")
      except:
        printWarning("Did not get U in "+tag)
      self.client.delete_tensor(tag+"U")
      # Account for Fortran/C memory layout and 32bit for TF
      data = np.transpose(data)
      data = np.expand_dims(data,axis=0)
      if do_init:
        state = data
        do_init = False
      else:
        state = np.append(state,data,axis=0)

    return state


  def _flexi_ended(self):
    """ Checks whether FLEXI has already ended."""
    has_ended = np.empty((self.n_envs))
    for i in range(self.n_envs):
      self.client.poll_tensor(self.tag[i]+"step_type",10,1000)
      step_type = self.client.get_tensor(self.tag[i]+"step_type")
      self.client.delete_tensor(self.tag[i]+"step_type")
      if step_type > 0:
        has_ended[i] = False
      else:
        has_ended[i] = True

    if np.all(has_ended):
      return True
    #elif np.all([not i for i in has_ended]):
    elif not np.all(has_ended):
      return False
    else:
      return None


  def _get_reward(self):
    """Compute the reward for the agent, based on the current flow state."""
    reward     = np.zeros( (self.n_envs,) )
    self.E_LES = np.zeros( (self.n_envs,self.reward_kmax) )
    for i in range(self.n_envs):
      # Poll Tensor
      self.client.poll_tensor(self.tag[i]+"Ekin",10,1000)
      data = self.client.get_tensor(self.tag[i]+"Ekin")
      self.client.delete_tensor(self.tag[i]+"Ekin")
      self.E_LES[i,:] = data[0:self.reward_kmax]

      # Compute Reward
      error  = self.E_LES[i,self.reward_kmin-1:self.reward_kmax] - self.E_DNS[self.reward_kmin-1:self.reward_kmax]
      error  = error/self.E_DNS[self.reward_kmin-1:self.reward_kmax]
      error  = np.square(error)
      reward[i] = 2.*np.exp(-1.*np.mean(error)/self.reward_scale)-1.

    return reward


  @property
  def can_plot(self):
    return MATPLOTLIB_FOUND


  def plot(self,kmax=None):
    """Wrapper routine for the plotting the energy spectra of the simulations"""
    if not MATPLOTLIB_FOUND:
      raise Exception("Matplotlib was not found but is necesasry to generate plots for Tensorboard!")

    if kmax:
      k_plot = kmax
    else:
      k_plot = self.reward_kmax

    return self._plot_spectra(self.E_DNS[:],self.E_LES[0,:],kmax=k_plot)


  def _plot_spectra(self,E_DNS,E_LES,kmax):
    """Plots the current LES and DNS energy spectra and returns the plot in a buffer"""

    # Plot Spectra
    fig = plt.figure(figsize=(6,5))
    k = range(1,kmax+1)
    plt.plot(k,E_DNS[0:kmax], label='DNS',linestyle='-',LineWidth=2)
    plt.plot(k,E_LES[0:kmax], label='LES',linestyle='-',LineWidth=2)
    plt.xlabel('k', fontsize=21)
    plt.ylabel(r'$E_{kin}$',fontsize=21)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1,kmax)
    plt.ylim(4.e-3,1.e0)
    plt.legend(fontsize=12)
    plt.grid(which='both')
    plt.tight_layout()

    # Buffer image as png in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Convert PNG buffer to TF image and add batch dimension
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image
