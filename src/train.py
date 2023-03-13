#!/usr/bin/env python3

import sys
import flexiEnvSmartSim
import models
import readin
import init_smartsim

from output import printWarning,printNotice
from output import printHeader,printBanner,printSmallBanner

import os
import io
import sys
import glob
import time
import random
import functools
import argparse
import subprocess
import shutil
import re
import copy
import contextlib

import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
#from absl import logging

from datetime import datetime

from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.system import multiprocessing
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver,dynamic_episode_driver
from tf_agents.policies import policy_saver
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import tf_py_environment,parallel_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer


# All flags registered in argparse must also be registered here!
FLAGS = flags.FLAGS
flags.DEFINE_integer('d', 0, help='Sets level of debug output. 0: Standard, 1: also Debug Output, 2: also FLEXI output.', lower_bound=0,upper_bound=2)
flags.DEFINE_integer('n', 1, 'Number of GPUs used', lower_bound=1)
flags.DEFINE_boolean('h', False, 'Print help')
flags.DEFINE_boolean('force-cpu', False, 'Forces TensorFlow to run on CPUs, even if GPUs are available.')


def write_metrics(metrics,step,category_name):
  """
  Adds all metrics in the list 'metrics' to the tf summary, which will be written to disk.
  'step' gives the corresponding global index and 'category_name' sets a prefix, which
  helps to group metrics in TensorBoard.
  """
  for metric in metrics:
    tf.summary.scalar(category_name+"/"+metric.name,metric.result().numpy(),step=step)


def init_gpus(num_gpus=1,gpu_memory_growth=False):
  """
  Check if tensorflow finds a GPU to run on and employs a distribution strategy
  on multi-GPU systems
  """
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      printNotice('Found '+str(len(gpus))+' physical GPU(s) on system.')
      # Set Memory growth
      if gpu_memory_growth:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True) # Allocate only neccessary memory

      # Check if enough GPUs available
      if (num_gpus > len(gpus)):
        printWarning('Requested more GPUs than available on the system. Use '+str(len(gpus))+' GPUs instead.')
        num_gpus = len(gpus)

      # Get Distribution Strategy
      if (num_gpus == 1):
        printNotice('Running on single GPU. To run on multiple GPUs use commandline argument "-n NUM_GPUS"')
        return None
      else:
        devices = []
        for i in range(num_gpus):
          devices.append("/gpu:"+str(i))

        printNotice('Running Mirrored Distribution Strategy on GPUs: '+",".join(devices))
        return tf.distribute.MirroredStrategy(devices=devices,cross_device_ops=tf.distribute.NcclAllReduce())

    except RuntimeError as e:
      printWarning(e) # Memory growth must be set before GPUs have been initialized
  else:
    printWarning('No GPU found on system, Tensorflow will probably run on CPU')
    return None


def parse_commandline_flags():
  """
  This routines parses the commandline options using the argparse library.
  ATTENTION: All flags registered in argparse must also be registered as flags to abseil, since abseil parses
             the flags first, and aborts if some flag is unknown.
             TODO: Fix this in the future!
  """
  parser = argparse.ArgumentParser(prog='python3 train.py'
                                  ,description='This is a Reinforcement Learning Framework for the DG solver FLEXI.'
                                  )
  parser.add_argument('--force-cpu'
                     ,default=False
                     ,dest='force_cpu'
                     ,action='store_true'
                     ,help='Forces TensorFlow to run on CPUs, even if GPUs are available.'
                     )
  parser.add_argument('-n'
                     ,metavar='N_GPU'
                     ,type=int
                     ,default=1
                     ,dest='num_gpus'
                     ,help='Number of GPUs used'
                     )
  parser.add_argument('-d'
                     ,metavar='DEBUG_LEVEL'
                     ,type=int
                     ,default=0
                     ,dest='debug'
                     ,help='Sets level of debug output. 0: Standard, 1: also Debug Output, 2: also FLEXI output.'
                     )
  parser.add_argument('config_file'
                     ,type=str
                     ,help='A configuration file in the YAML format.'
                     )

  # Only parse known flags. Unknown flags are later parsed by abseil
  args = parser.parse_args()

  return args

@tf.function
def distributed_train_step(agent,experience):
  return agent.train(experience).loss

def train_agent_distributed(agent,replay_buffer,strategy):
  with strategy.scope():
    # Get Dataset
    # TODO: Currently, we simply take all, thus num_steps and sample_batch_size have to simply be very large
    #       Implement this more cleanly
    dataset = replay_buffer.as_dataset(num_steps=50,sample_batch_size=512,single_deterministic_pass=True)
    # Set correct sharding policy to avoid tensorflow warnings
    dataset_options = tf.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset.with_options(dataset_options)
    # Generate distributed dataset from dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    # Get first element (we only have a single one...)
    dist_dataset_iterator = iter(dist_dataset)
    experience,_ = dist_dataset_iterator.get_next()
    # Train distributed
    per_replica_losses = strategy.run(distributed_train_step, args=(agent,experience,))
    #total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

  # Clear replay_buffer
  replay_buffer.clear()
  return


@tf.function
def train_agent(agent,replay_buffer):
  # Get Dataset
  dataset = replay_buffer.gather_all()
  # Train agent
  agent.train(experience=dataset)
  # Clear replay_buffer
  replay_buffer.clear()
  return


#@tf.function
def collect_trajectories(driver,env):
  # Startup FLEXI instances
  env.start()
  # Run FLEXI instances
  driver_run(driver)
  # Stop FLEXI instances
  env.stop()
  # Cleanup OMP files
  clean_ompi_tmpfiles()
  return

@tf.function
def driver_run(driver):
  final_time_step, policy_state = driver.run()

def generate_rankefile_hawk_ompi(hosts: list, cores_per_node: int, n_par_env: int, ranks_per_env: int, base_path=None):
  """Generate rank file for openmpi process binding
  :param host: list of hosts
  :type host: list[str]
  :param cores_per_node: number of cores per node
  :type cores_per_node: int
  :param n_par_env: number of parallel environments
  :type n_par_env: int
  :param ranks_per_env: number of ranks per environments
  :type ranks_per_env: int
  :param base_path: path to the base directory of the rank files
  :type base_path: str
  """

  # If no base_path given, use CWD
  if base_path:
    rankfile_dir = os.path.join(base_path, "ompi-rankfiles")
  else:
    rankfile_dir = "ompi-rankfiles"

  if os.path.exists(rankfile_dir):
    shutil.rmtree(rankfile_dir)
  os.makedirs(rankfile_dir, exist_ok=True)

  rankfiles = list()
  next_free_slot = 0
  n_cores_used = 0
  for env_idx in range(n_par_env):
    filename = os.path.join(rankfile_dir, f"par_env_{env_idx:05d}")
    rankfiles.append(filename)
    with open(filename, 'w') as rankfile:
      for i in range(ranks_per_env):
        rankfile.write(f"rank {i}={hosts[n_cores_used // cores_per_node]} slot={next_free_slot}\n")
        next_free_slot = next_free_slot + 1
        n_cores_used = n_cores_used + 1

        if next_free_slot > ( cores_per_node - 1 ):
          next_free_slot = 0

    files = os.listdir(rankfile_dir)

  return rankfiles


def parser_flexi_parameters(parameter_file, keyword, new_keyword_value):
  """
  """

  pattern = re.compile(r"(%s)\s*=.*" % keyword, re.IGNORECASE)
  subst = keyword + "=" + new_keyword_value
  parameter_file_in = parameter_file
  pbs_jobID = os.environ['PBS_JOBID']
  parameter_file_out = "parameter_flexi-" + pbs_jobID[0:7] + ".ini"

  with open(parameter_file_out,'w') as new_file:
    with open(parameter_file_in, 'r') as old_file:
      for line in old_file:
        new_file.write(pattern.sub(subst, line))

  return parameter_file_out


def clean_ompi_tmpfiles():
  """
  Cleans up temporary files which are created by openmpi in TMPDIR
  Avoids running out of space in TMPDIR
  If TMPDIR is not found exists with -1 status
  """
  try:
    tmpdir = os.environ['TMPDIR']
  except:
    return -1

  path = os.path.join(tmpdir,'ompi.*')
  path = glob.glob(path)

  for folder in path:
    for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      if os.path.isdir(file_path):
        shutil.rmtree(file_path)



def copy_to_nodes(my_files, base_path, hosts, subfolder=None):
  """
  This routine takes the files given in [my_files] and copies them
  to 'base_path' on the ssh targets 'hosts' via the scp command.
  If the path does not exists, it tries to create it via 'mkdir'.
  A optional 'subfolder' can be given, which will be appended to
  the 'base_path'.
  TODO: Implement a fail-safe, i.e. only overwite filepaths for
        which copying worked.
  """

  # If input not a list, i.e. a single element, transform into list
  if isinstance(my_files, list):
    conv_to_list = False
  else:
    my_files = [my_files]
    conv_to_list = True

  # Append subfolder if given
  if subfolder:
    target = os.path.join(base_path, subfolder)
  else:
    target = base_path

  # Copy to all given hosts
  for host in hosts:
    # Create folder if necessary
    os.system('ssh %s mkdir -p %s' % (host, target))
    # Copy files
    for my_file in my_files:
      os.system('scp -q "%s" "%s:%s"' % (my_file, host, target))

  # Get new path of files
  my_files_new = []
  for my_file in my_files:
    file_name = os.path.split(my_file)[1]
    my_files_new.append(os.path.join(target,file_name))

  # Convert back to single string if input is single string
  if conv_to_list:
    my_files_new = my_files_new[0]

  return my_files_new



def train( config_file
          ,parameter_file
          ,executable_path
          ,train_files
          ,train_learning_rate
          ,discount_factor
          ,reward_spectrum_file
          ,reward_scale
          ,reward_kmax
          ,mesh_file
          ,reward_kmin  = 1
          ,run_name     = None # Output directory will be named accordingly
          ,restart_dir  = None # Directory with checkpoints to restart from
          ,random_seed  = None
          ,log_interval = 1
          ,num_procs_per_environment  = 1
          ,num_parallel_environments  = 1
          #,num_episodes_per_iteration = 10
          ,train_num_epochs           = 5
          ,train_num_iterations       = 1000
          ,train_buffer_capacity      = 1000
          ,entropy_regularization     = 0.0
          ,importance_ratio_clipping  = 0.2
          ,action_std        = 0.02
          ,dist_type         = 'normal'
          ,ckpt_interval     = 5
          ,ckpt_num          = 1000
          ,eval_files        = None
          ,eval_num_episodes = 1
          ,eval_interval     = 5
          ,eval_write_states = False
          ,eval_do_analyze   = False
          ,use_XLA     = False
          ,do_profile  = False
          ,smartsim_port    = 6780
          ,smartsim_num_dbs = 1
          ,smartsim_launcher = "local"
          ,smartsim_orchestrator = "local"
          ,env_launcher = "mpirun"
          ,mpi_launch_mpmd = False
          ,local_dir = None
          ,n_procs_per_node=128 # Hawk
          ,strategy=None
          ,debug       = 0
        ):
  """
  Main training routine. Here, the (FLEXI) environment, the art. neural networks, the optimizer,...
  are instantiated and initialized. Then the main training loop iteratively collects trajectories
  and trains the agent on the sampled trajectories.
  """

  # Set output dir to restart dir, if given.
  # TODO: make this more robust
  if restart_dir is not None:
    base_dir = restart_dir

  # Otherwise, use run_name as output dir if given
  elif run_name is not None:
    base_dir = "logs/"+run_name

  # Else, use current time stamp
  else:
    base_dir = datetime.now().strftime('%d%m%y_%H%M%S')
    base_dir = "logs/"+base_dir

  # Set all output directories of the run accordingly
  train_dir = base_dir+"/train/"
  eval_dir  = base_dir+"/eval/"
  ckpt_dir  = base_dir+"/ckpt/"

  # Check if all necessary files actually exist
  missing_files = readin.files_exist([executable_path,parameter_file,train_files,eval_files,reward_spectrum_file])
  for item in missing_files:
    printWarning("The specified file "+item+" does not exist")

  # Activate XLA for performance
  if use_XLA:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
    tf.config.optimizer.set_jit(True)

  # Initialize SmartSim
  exp, worker_nodes, db, entry_db, is_db_cluster = init_smartsim.init_smartsim(port = smartsim_port
                                                                             ,num_dbs = smartsim_num_dbs
                                                                             ,launcher_type = smartsim_launcher
                                                                             ,orchestrator_type = smartsim_orchestrator
                                                                             )

  # generating rankfiles for OpenMPI
  if mpi_launch_mpmd:
    # If all MPI jobs are run with single mpirun command, all jobs are allocated based on single rankfile
    rank_files = generate_rankefile_hawk_ompi(worker_nodes
                                             ,n_procs_per_node
                                             ,n_par_env=1
                                             ,ranks_per_env=num_parallel_environments*num_procs_per_environment
                                             )

  else:
    # Otherwise every MPI job gets its own rankfile
    rank_files = generate_rankefile_hawk_ompi(worker_nodes
                                             ,n_procs_per_node
                                             ,num_parallel_environments
                                             ,num_procs_per_environment
                                             )

  # Copy all local files into local directory, possibly fast RAM-Disk or similar
  # for performance and to reduce Filesystem access
  if local_dir:
    # Prefix with PBS Job ID if PBS job
    if smartsim_launcher.casefold() == 'pbs':
      pbsJobID = os.environ['PBS_JOBID']
      local_dir = os.path.join(local_dir, pbsJobID)

    printNotice("Moving local files to %s ..." % (local_dir))

    # Get list of all nodes
    nodes = copy.deepcopy(worker_nodes)
    ai_node = os.environ['HOSTNAME']
    nodes.insert(0, ai_node)

    # Move all files to local dir
    # TODO: control which files are copied by 'local_files' variable!
    train_files          = copy_to_nodes(train_files,         local_dir,nodes,subfolder='train_files')
    eval_files           = copy_to_nodes(eval_files,          local_dir,nodes,subfolder='eval_files')
    reward_spectrum_file = copy_to_nodes(reward_spectrum_file,local_dir,nodes,subfolder='reward_files')
    rank_files           = copy_to_nodes(rank_files,          local_dir,nodes,subfolder='ompi_rank_files')
    mesh_file            = copy_to_nodes(mesh_file,           local_dir,nodes,subfolder='ompi_rank_files')

    # We have to update the meshfile in the parameter file before copying
    parameter_file = parser_flexi_parameters(parameter_file, 'MeshFile', mesh_file)
    parameter_file = copy_to_nodes(parameter_file,local_dir,nodes,subfolder='parameter_files')

    printNotice(" DONE! ",newline=False)

  if mpi_launch_mpmd:
    rank_files = [rank_files[0] for _ in range(num_parallel_environments)]


  # Instantiate parallel collection environment
  my_env = tf_py_environment.TFPyEnvironment(
           flexiEnvSmartSim.flexiEnv(exp
                                    ,executable_path
                                    ,parameter_file
                                    ,tag              = 'train'
                                    ,port             = smartsim_port
                                    ,entry_db         = entry_db
                                    ,is_db_cluster    = is_db_cluster
                                    ,hosts            = worker_nodes
                                    ,n_envs           = num_parallel_environments
                                    ,n_procs          = num_procs_per_environment
                                    ,n_procs_per_node = n_procs_per_node
                                    ,spectra_file     = reward_spectrum_file
                                    ,reward_kmin      = reward_kmin
                                    ,reward_kmax      = reward_kmax
                                    ,reward_scale     = reward_scale
                                    ,restart_files    = train_files
                                    ,rankfiles        = rank_files
                                    ,env_launcher     = env_launcher
                                    ,mpi_launch_mpmd  = mpi_launch_mpmd
                                    ,debug            = debug
                                    ))

  # Instantiate serial evaluation environment
  if eval_files is None:
    printWarning('No specific Files for Evaluation specified. Using Training files instead')
    eval_files = train_files

  my_eval_env = tf_py_environment.TFPyEnvironment(
                flexiEnvSmartSim.flexiEnv(exp
                                         ,executable_path
                                         ,parameter_file
                                         ,tag              = 'eval'
                                         ,port             = smartsim_port
                                         ,entry_db         = entry_db
                                         ,is_db_cluster    = is_db_cluster
                                         ,hosts            = worker_nodes
                                         ,n_procs          = num_procs_per_environment
                                         ,n_procs_per_node = n_procs_per_node
                                         ,spectra_file     = reward_spectrum_file
                                         ,reward_kmin      = reward_kmin
                                         ,reward_kmax      = reward_kmax
                                         ,reward_scale     = reward_scale
                                         ,restart_files    = eval_files
                                         ,random_restart_file = False
                                         ,rankfiles        = rank_files
                                         ,env_launcher     = env_launcher
                                         ,debug            = debug
                                         ))


  # Get training variables
  optimizer   = tf.keras.optimizers.Adam(learning_rate=train_learning_rate)
  global_step = tf.compat.v1.train.get_or_create_global_step()

  # For distribution strategy, networks and agent have to be initialized within strategy.scope
  if strategy:
    context = strategy.scope()
  else:
    context = contextlib.nullcontext() # Placeholder that does nothing

  with context:
    # Set TF random seed within strategy to obtain reproducible results
    if random_seed:
      random.seed(random_seed)        # Python seed
      tf.random.set_seed(random_seed) # TF seed

    # Instantiate actor net
    actor_net = models.ActionNetCNN(my_env.observation_spec()
                                     ,my_env.action_spec()
                                     ,action_std=action_std
                                     ,dist_type=dist_type
                                     ,debug=debug)
    value_net = models.ValueNetCNN( my_env.observation_spec()
                                     ,debug=debug)

    # PPO Agent
    tf_agent = ppo_clip_agent.PPOClipAgent(
            my_env.time_step_spec(),
            my_env.action_spec(),
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
            importance_ratio_clipping=importance_ratio_clipping,
            discount_factor=discount_factor,
            entropy_regularization=entropy_regularization,
            normalize_observations=False,
            normalize_rewards=False,
            use_gae=True,
            num_epochs=train_num_epochs,
            train_step_counter=global_step)
    tf_agent.initialize()

  # Get Agent's Policies
  eval_policy    = tf_agent.policy
  collect_policy = tf_agent.collect_policy

  # Instantiate Replay Buffer, which holds the sampled trajectories.
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                     data_spec  = tf_agent.collect_data_spec,
                     batch_size = my_env.batch_size,
                     max_length = train_buffer_capacity)

  # Currently sampling in several times not supported
  num_episodes_per_iteration=num_parallel_environments

  # Instantiate driver for data collection
  num_episodes = tf_metrics.NumberOfEpisodes(name='num_episodes')
  env_steps    = tf_metrics.EnvironmentSteps(name='num_env_steps')
  avg_return   = tf_metrics.AverageReturnMetric(  name='avg_return'
                                                 ,buffer_size=num_episodes_per_iteration
                                                 ,batch_size=num_parallel_environments)
  min_return   = tf_metrics.MinReturnMetric(      name='min_return'
                                                 ,buffer_size=num_episodes_per_iteration
                                                 ,batch_size=num_parallel_environments)
  max_return   = tf_metrics.MaxReturnMetric(      name='max_return'
                                                 ,buffer_size=num_episodes_per_iteration
                                                 ,batch_size=num_parallel_environments)
  train_metrics  = [num_episodes
                   ,env_steps
                   ,avg_return
                   ,min_return
                   ,max_return
                   ]

  collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(my_env
                                                              ,collect_policy
                                                              ,observers = train_metrics + [replay_buffer.add_batch]
                                                              ,num_episodes=num_episodes_per_iteration
                                                              )

  # Instantiate driver for evaluation
  eval_avg_eplen  = tf_metrics.AverageEpisodeLengthMetric(name='eval_avg_episode_length',buffer_size=eval_num_episodes)
  eval_avg_return = tf_metrics.AverageReturnMetric(       name='eval_avg_return'        ,buffer_size=eval_num_episodes)
  eval_metrics    = [eval_avg_eplen,eval_avg_return]
  eval_driver     = dynamic_episode_driver.DynamicEpisodeDriver(my_eval_env
                                                               ,eval_policy
                                                               ,observers = eval_metrics
                                                               ,num_episodes=eval_num_episodes
                                                               )
  # Write summary to TensorBoard
  summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=1000)

  # Define checkpointer to save policy
  train_checkpointer = common.Checkpointer(
                               ckpt_dir=ckpt_dir,
                               max_to_keep=ckpt_num,
                               agent=tf_agent,
                               policy=tf_agent.policy,
                               global_step=global_step)
  train_checkpointer.initialize_or_restore()


  # Main train loop
  printBanner('Starting Training Loop!')
  with summary_writer.as_default():
    if do_profile:
      printNotice('Starting profiling....')
      tf.profiler.experimental.start(train_dir)

    start_time = time.time()

    # In case of restart don't start counting at 0
    starting_iteration = int(global_step.numpy()/train_num_epochs)

    # Write parameter files to Tensorboard
    tf.summary.text("training_config"
                   ,readin.read_file(config_file,newline='  \n') # TF uses markdown EOL
                   ,step=starting_iteration)
    tf.summary.text("flexi_config"
                   ,readin.read_file(parameter_file,newline='  \n') # TF uses markdown EOL
                   ,step=starting_iteration)

    for i in range(starting_iteration ,train_num_iterations):

      if (i % eval_interval) == 0:
        mytime = time.time()
        collect_trajectories(eval_driver,my_eval_env)
        write_metrics(eval_metrics,global_step,'MetricsEval')

        # Plot Energy Spectra to Tensorboard
        if my_eval_env.can_plot:
          tf.summary.image("Spectra", my_eval_env.plot(), step=global_step)

        printNotice('Eval time: [%5.2f]s' % (time.time()-mytime))
        printNotice('Eval average return: %f' % (eval_avg_return.result().numpy()),newline=False)

      mytime = time.time()
      collect_trajectories(collect_driver,my_env)
      write_metrics(train_metrics,global_step,'MetricsTrain')
      collect_time = time.time()-mytime

      mytime = time.time()
      if strategy:
        train_agent_distributed(tf_agent,replay_buffer,strategy)
      else:
        train_agent(tf_agent,replay_buffer)
      train_time = time.time()-mytime

      # Log to console every log_interval iterations
      if (i % log_interval) == 0:
        printSmallBanner('ITERATION %i' % i)

        printNotice('Episodes:     %i' % (      num_episodes.result().numpy()),newline=False)
        printNotice('Env. Steps:   %i' % (         env_steps.result().numpy()),newline=False)
        printNotice('Train Steps:  %i' % (tf_agent.train_step_counter.numpy()),newline=False)

        printNotice('Collect time: [%5.2f]s' % (collect_time))
        printNotice('Train time:   [%5.2f]s' % (train_time)            ,newline=False)
        printNotice('TOTAL:        [%5.2f]s' % (time.time()-start_time),newline=False)

      # Checkpoint the policy every ckpt_interval iterations
      if (i % ckpt_interval) == 0:
        printNotice('Saving checkpoint to: ' + ckpt_dir)
        train_checkpointer.save(global_step)
        #tf_policy_saver.save(ckpt_dir)

      # Flush summary to TensorBoard
      tf.summary.flush()

    if do_profile:
      printNotice('End profiling.')
      tf.profiler.experimental.stop()

    # Close all
    del my_env
    del my_eval_env

    exp.stop(db)
    time.sleep(2.) # Wait for orchestrator to be properly closed

    # Finalize Training
    printBanner('Sucessfully finished after: [%8.2f]s' % (time.time()-start_time))



def main(argv):
  # Parse Commandline arguments
  args = parse_commandline_flags()

  # Print Header to Console
  printHeader()

  # Check if we should run on CPU or GPU
  if args.force_cpu: # Force CPU execution
    os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
    printWarning('TensorFlow will be forced to run on CPU')
    strategy = None
  else: # Check if we find a GPU to run on
    strategy = init_gpus(args.num_gpus)

  # Set TF logging level
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(2-args.debug) # in TF higher value means less output

  # Set SmartSim logging level
  if args.debug == 0:
    os.environ["SMARTSIM_LOG_LEVEL"] = "quiet" # Only Warnings and Errors
  elif args.debug == 1:
    os.environ["SMARTSIM_LOG_LEVEL"] = "info"  # Additional Information
  elif args.debug == 2:
    os.environ["SMARTSIM_LOG_LEVEL"] = "debug" # All available Output

  # Parse Config file
  config = readin.read_config(args.config_file)

  # Start training with the parameters of config, which are passed to the function as dict.
  train(debug=args.debug,config_file=args.config_file,**config,strategy=strategy)


if __name__ == "__main__":
  # Multiprocessing wrapper of main function from tf-agent
  multiprocessing.handle_main(functools.partial(app.run, main))
