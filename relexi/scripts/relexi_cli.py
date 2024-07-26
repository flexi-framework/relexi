#!/usr/bin/env python3

"""Wrapper to launch training routine."""

import os
import time
import argparse

import relexi
import relexi.io.output as rlxout
import relexi.io.readin as rlxin
import relexi.rl.ppo as rlxppo
from relexi.rl.tf_helpers import init_gpus


def parse_commandline_flags():
    """Parse commandline options with `argparse`.

    This routines defines the commnadline arguments and parses them via the
    `argparse` library.

    Return:
        args: Parsed commandline arguments.
    """
    parser = argparse.ArgumentParser(
        prog='relexi',
        description='Relexi: A reinforcement learning library for simulation environments on high-performance computing systems.'
    )
    parser.add_argument(
            '-v',
            '--version',
            action='version',
            version=f'%(prog)s {relexi.__version__}'
            )
    parser.add_argument(
            '--force-cpu',
            default=False,
            dest='force_cpu',
            action='store_true',
            help='Forces TensorFlow to run on CPUs.'
            )
    parser.add_argument(
            '-n',
            metavar='N_GPU',
            type=int,
            default=1,
            dest='num_gpus',
            help='Number of GPUs used'
            )
    parser.add_argument(
            '-d',
            metavar='DEBUG_LEVEL',
            type=int,
            default=0,
            dest='debug',
            help='Output level. 0: Standard, 1: also Debug Output, 2: also FLEXI output.'
            )
    parser.add_argument(
            'config_file',
            type=str,
            help='A configuration file in the YAML format.'
            )

    # Only parse known flags. Unknown flags are later parsed by abseil
    args = parser.parse_args()

    return args


def main():
    """Parses arguments and starts training."""
    start_time = time.time()

    args = parse_commandline_flags()

    # Print Header to Console
    rlxout.header()

    # Check if we should run on CPU or GPU
    if args.force_cpu:  # Force CPU execution
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        rlxout.warning('TensorFlow will be forced to run on CPU')
        strategy = None
    else:  # Check if we find a GPU to run on
        strategy = init_gpus(args.num_gpus)

    # Set TF logging level (in TF, higher value means less output)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(2 - args.debug)

    # Set SmartSim logging level
    if args.debug == 0:
        os.environ["SMARTSIM_LOG_LEVEL"] = "quiet"  # Only Warnings and Errors
    elif args.debug == 1:
        os.environ["SMARTSIM_LOG_LEVEL"] = "info"   # Additional Information
    elif args.debug == 2:
        os.environ["SMARTSIM_LOG_LEVEL"] = "debug"  # All available Output

    # Parse Config file
    config = rlxin.read_config(args.config_file)

    # Start training the parameters of config (passed to the function as dict)
    rlxppo.train(
        debug=args.debug,
        config_file=args.config_file,
        **config,
        strategy=strategy
    )

    # Provide final banner
    rlxout.banner(f'Sucessfully finished after: [{time.time()-start_time:8.2f}]s')


if __name__ == "__main__":
    main()
