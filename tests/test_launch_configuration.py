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


"""Test for the launchconfig module based on `unittest` package."""

import unittest

from .context import relexi
from relexi.runtime import LaunchConfig, Runtime

class TestLaunchConfig(unittest.TestCase):
    """Tests for the launchconfig module."""

    #@unittest.mock.patch('runtime.Runtime.n_worker_slots', return_value=4)
    #def test_launchconfig_init_local(self):
    #    """Test init of launchconfig with database."""
    #    launchconfig = LaunchConfig(type_='local')
    #    assert launchconfig is not None
    #    assert launchconfig.type == 'local'

    def test_generate_rankfile_ompi(self):
        """Test generate_rankfile_ompi."""
        # Specify configuration
        workers = ['r1n1c1n1', 'r1n1c1n2']
        n_slots_per_worker = 4
        n_par_env = 4
        n_procs = [2, 2, 1, 3]
        # Generate rankfiles
        rankfiles = LaunchConfig._generate_rankfile_ompi(
                                                         workers,
                                                         n_slots_per_worker,
                                                         n_par_env,
                                                         n_procs
                                                         )
        # Prepare expected content
        expected = [
            'rank 0=r1n1c1n1 slot=0\nrank 1=r1n1c1n1 slot=1',
            'rank 0=r1n1c1n1 slot=2\nrank 1=r1n1c1n1 slot=3',
            'rank 0=r1n1c1n2 slot=0',
            'rank 0=r1n1c1n2 slot=1\nrank 1=r1n1c1n2 slot=2\nrank 2=r1n1c1n2 slot=3'
        ]
        # Check that rankfiles are correct
        for i, rankfile in enumerate(rankfiles):
            with open(rankfile, 'r', encoding='utf-8') as f:
                file_lines = f.read().rstrip()
                print(f'Rankfile {i}:\n{file_lines}')
                assert file_lines == expected[i]

    def test_distribute_workers_slurm_1(self):
        """Test distribute_workers_slurm."""
        # Specify configuration
        n_procs = [2, 2, 1, 3]
        n_exe = 4
        workers = ['r1n1c1n1', 'r1n1c1n2']
        procs_avail = 8
        # Distribute workers
        hosts_per_exe = LaunchConfig._distribute_workers_slurm(
                                                        n_procs,
                                                        n_exe,
                                                        workers,
                                                        procs_avail
                                                        )
        # Check that workers are correct
        expected = [['r1n1c1n1'], ['r1n1c1n1'], ['r1n1c1n2'], ['r1n1c1n2']]
        assert expected == hosts_per_exe

    def test_distribute_workers_slurm_2(self):
        """Test distribute_workers_slurm."""
        # Specify configuration
        n_procs = [3, 3]
        n_exe = 2
        workers = ['r1n1c1n1', 'r1n1c1n2']
        procs_avail = 8
        # Distribute workers
        hosts_per_exe = LaunchConfig._distribute_workers_slurm(
                                                        n_procs,
                                                        n_exe,
                                                        workers,
                                                        procs_avail
                                                        )
        # Check that workers are correct
        expected = [['r1n1c1n1'], ['r1n1c1n2']]
        assert expected == hosts_per_exe

    def test_distribute_workers_slurm_3(self):
        """Test errors raised when not sufficient resources."""
        # Specify configuration
        n_procs = [3, 3]
        n_exe = 2
        workers = ['r1n1c1n1', 'r1n1c1n2']
        procs_avail = 4
        # Check that workers are correct
        self.assertRaises(
                          RuntimeError,
                          LaunchConfig._distribute_workers_slurm,
                          n_procs,
                          n_exe,
                          workers,
                          procs_avail
                          )
