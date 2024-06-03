#!/usr/bin/env python3

"""Contains helper functions for managing simulations on HPC systems.

This module implements several auxiliary functions to manage simulation
instances on HPC systems. This contains in particular functions to generate
rankfiles for OpenMPI for correct mappings and to move simulation files to the
worker nodes for faster startup.
"""

import os
import re
import glob
import shutil


def generate_rankfile_ompi(hosts, cores_per_node, n_par_env, ranks_per_env, base_path=None):
    """Generate rank file for OpenMPI process binding.

    Args: 
        hosts (list): List of hostnames
        cores_per_node (int): Number of cores per node
        n_par_env (int): Number of parallel environments to be launched
        ranks_per_env (int): Number of ranks per environments
        base_path (str): (Optional.) Path to the directory of the rank files

    Returns:
        list: List of filenames of the rankfiles
    """

    # If no base_path given, use CWD
    if base_path:
        rankfile_dir = os.path.join(base_path, "ompi-rankfiles")
    else:
        rankfile_dir = "ompi-rankfiles"

    if os.path.exists(rankfile_dir):
        shutil.rmtree(rankfile_dir)
    os.makedirs(rankfile_dir, exist_ok=True)

    rankfiles = []
    next_free_slot = 0
    n_cores_used = 0
    for env_idx in range(n_par_env):
        filename = os.path.join(rankfile_dir, f"par_env_{env_idx:05d}")
        rankfiles.append(filename)
        with open(filename, 'w', encoding='ascii') as rankfile:
            for i in range(ranks_per_env):
                rankfile.write(f"rank {i}={hosts[n_cores_used//cores_per_node]} slot={next_free_slot}\n")
                next_free_slot = next_free_slot + 1
                n_cores_used = n_cores_used + 1
                if next_free_slot > (cores_per_node - 1):
                    next_free_slot = 0
    return rankfiles


def parser_flexi_parameters(parameter_file, keyword, value):
    """Changes the value for a keyword in a FLEXI parameter file.

    The FLEXI parameter file is structured as in the form of
        `keyword = value`
    This function changes the `value` for a given keyword using regular
    expressions and writes the modified contents to a new file.

    Args:
        parameter_file (str): Path to FLEXI parameterfile.
        keyword (str): Keyword of the parameter.
        value (str): New value for the given keyword.

    Returns:
        str: Path to new (modified) parameter file
    """
    pattern = re.compile(fr"({keyword})\s*=.*", re.IGNORECASE)
    subst = keyword + "=" + value
    parameter_file_in = parameter_file
    pbs_job_id = os.environ['PBS_JOBID']
    parameter_file_out = f"parameter_flexi-{pbs_job_id[0:7]}.ini"

    with open(parameter_file_out, 'w', encoding='ascii') as new_file:
        with open(parameter_file_in, 'r', encoding='ascii') as old_file:
            for line in old_file:
                new_file.write(pattern.sub(subst, line))
    return parameter_file_out


def clean_ompi_tmpfiles(env_variable="TMPDIR"):
    """Cleans up temporary files created by OpenMPI.

    OpenMPI creates temporary files with each invocation, which might cause the
    system to run out of disk space at some point. This routine tries to locate
    the current folder for temporary files by evaluating the env variable
    $TMPDIR and tries to remove all OpenMPI-specific files it finds.

    Args:
        env_variable (string): Name for environment variable where current
            folder for termporary files is stored.

    Returns:
        int:
            * 1 if operation was successfull,
            * -1 otherwise.
    """
    try:
        tmpdir = os.environ[env_variable]
    except RuntimeError:
        return -1

    path = os.path.join(tmpdir, 'ompi.*')
    path = glob.glob(path)

    for folder in path:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isdir(file_path):
                try:
                    shutil.rmtree(file_path)
                except RuntimeError:
                    return -1
    return 1


def copy_to_nodes(my_files, base_path, hosts, subfolder=None):
    """Copy files to local disk storage on allocated nodes.

    This routine takes the files given in `my_files` and copies them to
    `base_path` on the ssh targets `hosts` via the `scp` command. If the path
    does not exists, it tries to create it via `mkdir`. An optional `subfolder`
    can be given, which will be appended to the `base_path`.

    Args:
        my_files (list): List of path names to the files that are copied.
        base_path (list): Folder on hosts to which the files should be copied.
        hosts (list): List of hosts that are the ssh targets.
        subfolder (str): (Optional.)

    Return:
        list: List of paths to files on hosts.

    TODO:
        Implement a fail-safe: only overwrite filepaths for which copying
        worked.
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
        os.system(f'ssh {host} mkdir -p {target}')
        # Copy files
        for my_file in my_files:
            os.system(f'scp -q "{my_file}" "{host}:{target}"')

    # Get new path of files
    my_files_new = []
    for my_file in my_files:
        file_name = os.path.split(my_file)[1]
        my_files_new.append(os.path.join(target, file_name))

    # Convert back to single string if input is single string
    if conv_to_list:
        my_files_new = my_files_new[0]

    return my_files_new
