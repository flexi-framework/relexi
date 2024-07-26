#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Relexi, a reinforcement learning framework for training
# machine learning models in simulations on high-performance computing systems.
#
# Copyright (c) 2022-2024 Marius Kurz, Andrea Beck
#
# Relexi is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Relexi is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Relexi. If not, see <http://www.gnu.org/licenses/>.


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
