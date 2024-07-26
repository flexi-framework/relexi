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


"""Tests for the runtime module based on `unittest` package."""

import os
import socket
import unittest
from unittest import mock

import smartsim

from .context import relexi
from relexi.runtime import Runtime


class TestRuntime(unittest.TestCase):
    """Tests for the runtime module."""
    def test_runtime_auto(self):
        """Test auto-detection of runtime type."""
        runtime = Runtime(type_='auto', do_launch_orchestrator=False)
        assert runtime is not None
        assert runtime.type == smartsim.wlm.detect_launcher()

    def test_runtime_init(self):
        """Test init of runtime with database."""
        runtime = Runtime(type_='local')
        assert runtime is not None
        assert runtime.type == 'local'

    def test_runtime_info_local(self):
        """For Info, we just want to check if it runs without errors."""
        try:
            runtime = Runtime(type_='local')
            runtime.info()
        except Exception as e:
            self.fail(f"Runtime.info() raised an exception: {e}")

    def test_runtime_env_local(self):
        runtime = Runtime(type_='local', db_port=6780)
        assert runtime is not None
        assert runtime.type == 'local'
        assert not runtime.is_distributed
        assert runtime.hosts == [runtime.head]
        assert runtime.hosts == runtime.workers
        assert runtime.db_entry == '127.0.0.1:6780'
        assert runtime.db is not None
        assert runtime.exp is not None

    @mock.patch.dict(os.environ, {'SLURM_JOB_CPUS_PER_NODE': '4(x2),8,16(x2)'})
    def test_runtime_init_slurm(self):
        """Test setup based on mocked SLURM_JOB_CPUS_PER_NODE."""
        # Also mock retrieval of hostnames, since smartsim util needs
        # `scontrol` for that, which is not installed on non-SLURM systems.
        with mock.patch('smartsim.wlm.slurm.get_hosts',
                        return_value=['node1', 'node2', 'node3', 'node4', 'node5']):
          # Set localhost to correct name
          with mock.patch('socket.gethostname', return_value='node1'):
              runtime = Runtime(type_='slurm', do_launch_orchestrator=False)
              assert runtime is not None
              assert runtime.is_distributed
              assert runtime.type == 'slurm'
              assert runtime.hosts == ['node1', 'node2', 'node3', 'node4', 'node5']
              assert runtime.workers == ['node2','node3', 'node4', 'node5']
              assert runtime.head == 'node1'
              assert runtime.n_worker_slots == 44
              assert runtime._get_slots_per_node_slurm() == [4, 4, 8, 16, 16]

    @mock.patch.dict(os.environ, {'PBS_NODEFILE': '.nodefile.mock'})
    def test_runtime_env_pbs_1(self):
        """Test setup based on mocked nodefile."""
        # Prepare rank file
        with open('.nodefile.mock', 'w', encoding='utf-8') as f:
            f.write('node1\nnode2\nnode2\nnode2\nnode3\nnode3\n')
        # Set localhost to correct name
        with mock.patch('socket.gethostname', return_value='node1'):
            runtime = Runtime(type_='pbs', do_launch_orchestrator=False)
            assert runtime is not None
            assert runtime.is_distributed
            assert runtime.type == 'pbs'
            assert runtime.hosts == ['node1', 'node2', 'node3']
            assert runtime.workers == ['node2','node3']
            assert runtime.head == 'node1'
            assert runtime.n_worker_slots == 5
            assert runtime._get_slots_per_node_pbs() == [1, 3, 2]

    @mock.patch.dict(os.environ, {'PBS_NODEFILE': '.nodefile.mock'})
    def test_runtime_env_pbs_2(self):
        """Test long-form hostnames in nodefile."""
        # Prepare rank file
        with open('.nodefile.mock', 'w', encoding='utf-8') as f:
            f.write('node1.some.thing\nnode1.some.thing\nnode2.some.thing\nnode2.some.thing\nnode2.some.thing\nnode3.some.thing\n')
        # Set localhost to correct name
        with mock.patch('socket.gethostname', return_value='node1'):
            runtime = Runtime(type_='pbs', do_launch_orchestrator=False)
            assert runtime is not None
            assert runtime.is_distributed
            assert runtime.type == 'pbs'
            assert runtime.hosts == ['node1', 'node2', 'node3']
            assert runtime.workers == ['node2','node3']
            assert runtime.head == 'node1'
            assert runtime.n_worker_slots == 4
            assert runtime._get_slots_per_node_pbs() == [2, 3, 1]

    @mock.patch.dict(os.environ, {'PBS_NODEFILE': '.nodefile.mock'})
    def test_runtime_env_pbs_3(self):
        """Test fallback to 'local' mode for empty nodefile"""
        # Prepare rank file
        with open ('.nodefile.mock', 'w', encoding='utf-8') as f:
            f.write('')
        runtime = Runtime(type_='pbs', do_launch_orchestrator=False)
        assert runtime is not None
        assert runtime.type == 'local'
        assert not runtime.is_distributed
        assert runtime.hosts == [runtime.head]
        assert runtime.hosts == runtime.workers

    def test_runtime_init_wrong(self):
        """Test fallback for invalid runtime type"""
        runtime = Runtime(type_='wrong')
        assert runtime.type == 'local'
        assert not runtime.is_distributed
        assert runtime.hosts == [runtime.head]
        assert runtime.hosts == runtime.workers
        assert runtime.db is not None
        assert runtime.exp is not None
