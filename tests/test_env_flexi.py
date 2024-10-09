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


import os.path
import unittest
from unittest.mock import patch

from smartsim import Experiment

from .context import relexi
import relexi.env.hit.flexiEnvSmartSim as rlxenv

""" Contains pytest - tests for the functionalities of the relexi.env.flexiEnv module """

@patch.object(rlxenv.flexiEnv,'_start_flexi')
@patch.object(rlxenv.flexiEnv,'_get_current_state')
@patch.object(rlxenv.flexiEnv,'stop')
@patch('rlxenv.Client')
def init_flexi_env(mock__start_flexi, mock__get_current_state, mock_stop, mock_Client):

    smartsim_port = 6780
    smartsim_num_dbs = 1
    smartsim_launcher = "local"
    #orchestrator_type = "local"
    num_parallel_environments = 8
    num_procs_per_environment = 8
    n_procs_per_node = 16
    reward_kmin = 1
    reward_kmax=9
    reward_scale=0.4
    train_files = ['./simulation_files/run_f200_N5_4Elems_State_0000003.000000000.h5'
                  ,'./simulation_files/run_f200_N5_4Elems_State_0000004.000000000.h5'
                  ,'./simulation_files/run_f200_N5_4Elems_State_0000005.000000000.h5'
                  ,'./simulation_files/run_f200_N5_4Elems_State_0000006.000000000.h5'
                  ]
    rank_files = ['/my/dummy/rankfile_00'
                 ,'/my/dummy/rankfile_01'
                 ,'/my/dummy/rankfile_02'
                 ,'/my/dummy/rankfile_03'
                 ,'/my/dummy/rankfile_04'
                 ,'/my/dummy/rankfile_05'
                 ,'/my/dummy/rankfile_06'
                 ,'/my/dummy/rankfile_07'
                 ]
    env_launcher = 'mpirun'
    mpi_launch_mpmd = False
    debug = 0
    smartsim_orchestrator = "local"
    smartsim_launcher = "local"
    entry_db="127.0.0.1"
    worker_nodes = ["r1c1t1n1","r1c1t1n2","r1c1t1n3","r1c1t1n4"]
    reward_spectrum_file = "./examples/HIT_24_DOF/simulation_files/DNS_spectrum_stats_t2_to_t10.csv"
    parameter_file = "parameter.ini"
    executable_path = "/flexi-extensions/build/bin/flexi"
    is_db_cluster = False

    exp = Experiment("flexi", launcher=smartsim_launcher)

    flexi_env = rlxenv.flexiEnv(exp
                               ,executable_path
                               ,parameter_file
                               ,tag              = 'eval'
                               ,port             = smartsim_port
                               ,entry_db         = entry_db
                               ,hosts            = worker_nodes
                               ,n_procs          = num_procs_per_environment
                               ,n_envs           = num_parallel_environments
                               ,n_procs_per_node = n_procs_per_node
                               ,spectra_file     = reward_spectrum_file
                               ,reward_kmin      = reward_kmin
                               ,reward_kmax      = reward_kmax
                               ,reward_scale     = reward_scale
                               ,restart_files    = train_files
                               ,random_restart_file = False
                               ,rankfiles        = rank_files
                               ,env_launcher     = env_launcher
                               ,debug            = debug
                               )

    return flexi_env

@unittest.skip('Has to be adapted to new Runtime implementation.')
@patch('os.path.isfile')
@patch('os.access')
@patch.object(Experiment, 'start')
def test_envFlexi_start_flexi(mock_isfile, mock_access, mock_start):

    n_envs_expected = 8
    exe_expected = "/flexi-extensions/build/bin/flexi"
    exe_args_expected = ['parameter.ini','./simulation_files/run_f200_N5_4Elems_State_0000003.000000000.h5', '--tag', 'eval']
    run_args_expected = {'rankfile': '/my/dummy/rankfile_0', 'report-bindings': '', 'n': 8}

    flexi_env = init_flexi_env()
    flexi = flexi_env._start_flexi(flexi_env.exp, flexi_env.n_procs, flexi_env.n_envs )

    n_envs = len(flexi)
    assert n_envs == n_envs_expected, f"Wrong number of models"

    j = 0
    k = 1
    for model in flexi:
        assert model.run_settings.exe[0] == exe_expected

        i = 0
        for exe_arg_expected in exe_args_expected:
            if exe_arg_expected == "eval":
                exe_arg_expected = exe_arg_expected + f"{j}_"
            assert model.run_settings.exe_args[i] == exe_arg_expected
            i=i+1

        for key, run_arg_expected in run_args_expected.items():
            if run_arg_expected == "/my/dummy/rankfile_0":
                run_arg_expected = run_arg_expected + f"{j}"
            if run_arg_expected == "r1c1t1n":
                if j % 2 == 0:
                    run_arg_expected = run_arg_expected + f"{k}"
                else:
                    run_arg_expected = run_arg_expected + f"{k}"
                    k=k+1
            assert model.run_settings.run_args[key] == run_arg_expected

        j=j+1
