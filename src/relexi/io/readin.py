#!/usr/bin/env python3

import os
import yaml

"""
This module contains some helper functions for reading files,
especially the YAML-based configuration file.
"""

def read_config(config_file,flatten=True):
  """
  Readin the YAML-file config_filei. If flatten=True, remove uppermost level keys.
  """
  with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

  if flatten:
    return flatten_dict(config)
  else:
    return config


def flatten_dict(dict_in):
  """
  Remove the uppermost level of keys by concatenating the individual dicts of
  every key in the uppermost level.
  """
  dict_out={}
  for key in dict_in.keys():
    dict_out.update(dict_in[key])
  return dict_out


def flatten_list(list_in):
  """
  Recusively converts a nested list of arbitrary depth to a concatenated 1D list.
  """
  list_out = []

  for item in list_in:
    if isinstance(item,list):
      list_out.extend(flatten_list(item))
    else:
      list_out.append(item)

  return list_out


def read_file(filename,newline=None):
  """
  Reads a text file 'filename' and dumps all lines into a single string, while
  retaining the newline characters. If present, the newline character '\n' can
  be replaced by the string "newline", if specified.
  """
  with open(filename,'r') as myfile:
    data = myfile.read()

  if newline:
    return data.replace('\n', newline)
  else:
    return data


def files_exist(files):
  """
  Checks whether all files in the (nested) list 'files' actually exist and
  returns a list of all files, which do not exist.
  """
  # Flatten list to single dimension
  files_flatten = flatten_list(files)

  missing_elements = []
  for item in files_flatten:
    if not os.path.isfile(item):
      missing_elements.append(item)

  return missing_elements
