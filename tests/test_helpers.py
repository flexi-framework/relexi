#!/usr/bin/env python3

from .context import relexi
import relexi.smartsim.helpers as rlxhelpers

""" Contains pytest - tests for the functionalities of the relexi.smartsim.helpers module """

def test_generate_rankefile_hawk_ompi(tmp_path):
    
    hosts = ["r1n1c1n1", "r1n1c1n2"]
    cores_per_node = 4
    n_par_env = 4
    ranks_per_env = 2
    base_path = tmp_path

    expected = list()
    
    expected.append("rank 0=r1n1c1n1 slot=0\nrank 1=r1n1c1n1 slot=1")
    expected.append("rank 0=r1n1c1n1 slot=2\nrank 1=r1n1c1n1 slot=3")
    expected.append("rank 0=r1n1c1n2 slot=0\nrank 1=r1n1c1n2 slot=1")
    expected.append("rank 0=r1n1c1n2 slot=2\nrank 1=r1n1c1n2 slot=3")

    rankfiles_out = rlxhelpers.generate_rankefile_hawk_ompi(hosts, cores_per_node, n_par_env, ranks_per_env, base_path)
    
    i = 0
    for rankfile in rankfiles_out:
        with open(rankfile, 'r') as fh:
            assert fh.read().rstrip()==expected[i], f"Rankfile for rank {i} is wrong"
        i = i+1
