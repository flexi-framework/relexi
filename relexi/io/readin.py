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


"""Contains helper functions for reading files.

This module implements several auxiliary functions to handle the YAML-based
configuration file for Relexi.
"""

import os
import yaml


def read_config(file_in, flatten=True):
    """Read a YAML-file.

    Args:
        file_in (str): Path to the file to read.
        flatten (bool): (Optional.) Flag to remove uppermost hierarchy of keys.

    Returns:
        dict: Dictionary containing the contents of the file
    """
    with open(file_in, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    if flatten:
        return flatten_dict(config)
    return config


def flatten_dict(dict_in):
    """Converts nested dictionary to concatenated dictionary.

    Arguments:
        dict_in (dict): Possibly nested dictionary that will be flattened to
            single level of hierarchy.

    Returns:
        dict: Returns the flattened dictionary.
    """
    dict_out = {}

    for key, item in dict_in.items():
        if isinstance(item, dict):
            dict_out.update(flatten_dict(dict_in[key]))
        else:
            dict_out[key] = dict_in[key]

    return dict_out


def flatten_list(list_in):
    """Converts nested list of arbitrary depth to concatenated 1D list.

    Arguments:
        list_in (list): Possibly nested list which will be flattened to a
            concatenated 1D list.

    Returns:
        list: Returns the flattened 1D list.
    """
    list_out = []
    for item in list_in:
        if isinstance(item, list):
            list_out.extend(flatten_list(item))
        else:
            list_out.append(item)
    return list_out


def read_file(filename, newline=None):
    r"""Reads a text file and dumps all lines into a single string.

    Reads a text file and dumps all lines into a single string, while retaining
    the newline characters. If present, the newline character `\n` can be
    replaced by the string `"newline"`, if specified.

    Args:
        filename (str): Path to file that is read (relative or absolute).
        newline (str): (Optional.) All occurences of `\n` will be replaced with
            this string, if specified.

    Returns:
        str: Returns single string with the content of the file.
    """
    with open(filename, 'r', encoding='utf-8') as myfile:
        data = myfile.read()
    if newline:
        return data.replace('\n', newline)
    return data


def files_exist(files):
    """Check whether list of files exist.

    Checks whether all files with their paths specified in a (nested) list
    actually exist amd returns a list of all files, which do not exist.

    Args:
        files (list): Nested list of filepaths.

    Returns:
        list: List of filepaths that could not be found. Returns an empty list,
            if all files were found.
    """
    # Flatten list to single dimension
    files_flatten = flatten_list(files)

    missing_elements = []
    for item in files_flatten:
        if not os.path.isfile(item):
            missing_elements.append(item)
    return missing_elements
