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


import relexi.io.readin as rlxin

""" Contains pytest - tests for the functionalities of the relexi.io.readin module """

def test_flatten_dict():
  my_dict_in   = {"key1":{"key2":2, "key3":3}, "key4":{"key5":5, "key6":{"key7":7, "key8":8}},"key9":9}
  my_dict_out  = rlxin.flatten_dict(my_dict_in)
  my_dict_test = {"key2":2, "key3":3, "key5":5, "key7":7, "key8":8, "key9":9}
  assert my_dict_out==my_dict_test

def test_flatten_list():
  my_list_in   = ["key1",["key2", "key3"], "key4", ["key5", "key6",["key7", "key8"]],"key9"]
  my_list_out  = rlxin.flatten_list(my_list_in)
  my_list_test = ["key1","key2","key3","key4","key5","key6","key7","key8","key9"]
  assert my_list_out==my_list_test

if __name__ == "__main__":
  test_flatten_dict()
  test_flatten_list()
