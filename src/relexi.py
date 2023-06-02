#!/usr/bin/env python3

import relexi.io.output as rlxout
import relexi.io.readin as rlxin
import relexi.rl.ppo    as rlxppo
from relexi.rl.tf_helpers import init_gpus
from tf_agents.system import multiprocessing

import os
import time
import functools
import argparse

from absl import app
from absl import flags
#from absl import logging

from datetime import datetime

# All flags registered in argparse must also be registered here!
FLAGS = flags.FLAGS
flags.DEFINE_integer('d', 0, help='Sets level of debug output. 0: Standard, 1: also Debug Output, 2: also FLEXI output.', lower_bound=0,upper_bound=2)
flags.DEFINE_integer('n', 1, 'Number of GPUs used', lower_bound=1)
flags.DEFINE_boolean('h', False, 'Print help')
flags.DEFINE_boolean('force-cpu', False, 'Forces TensorFlow to run on CPUs, even if GPUs are available.')


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


def main():
  # Parse Commandline arguments
  args = parse_commandline_flags()

  # Print Header to Console
  rlxout.printHeader()

  # Check if we should run on CPU or GPU
  if args.force_cpu: # Force CPU execution
    os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
    rlxout.printWarning('TensorFlow will be forced to run on CPU')
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
  config = rlxin.read_config(args.config_file)

  # Start training with the parameters of config, which are passed to the function as dict.
  rlxppo.train(debug=args.debug,config_file=args.config_file,**config,strategy=strategy)

  # Finalize Training
  rlxout.printBanner('Sucessfully finished after: [%8.2f]s' % (time.time()-start_time))


if __name__ == "__main__":
  # Multiprocessing wrapper of main function from tf-agent
  #multiprocessing.handle_main(functools.partial(app.run, main))
  main()
