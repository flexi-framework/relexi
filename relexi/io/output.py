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


"""Contains helper functions to harmonize the output to console."""

STD_LENGTH = 80
"""Standard length for output to console."""


class Colors:
    """Print big header with program name and logo to console.

    Attributes:
        BANNERA (str): Defines first banner color for Header and large banners.
        BANNERB (str): Defines second banner color for small banners.
        WARN    (str): Defines color for warnings.
        END     (str): Defines end of color in string.
    """
    BANNERA = '\033[93m'
    BANNERB = '\033[94m'
    WARN = '\033[91m'
    END = '\033[0m'


def header(length=STD_LENGTH):
    """Print big header with program name and logo to console.

    Args:
        length (int): Number of characters used within each line.
    """
    print(Colors.BANNERA+'='*length)
    print('')
    print(Colors.BANNERA + " 000000000000    000000000000  000          000000000000  00000     00000  000")
    print(Colors.BANNERA + " 0000000000000   000000000000  000          000000000000   00000   00000   000")
    print(Colors.BANNERA + " 000        000  000           000          000              000000000     000")
    print(Colors.BANNERA + " 000        000  000  0000000  000          000  0000000       00000       000")
    print(Colors.BANNERA + " 000  00000000   000  0000000  000          000  0000000      0000000      000")
    print(Colors.BANNERA + " 000  0000000    000           000          000              0000 0000     000")
    print(Colors.BANNERA + " 000      0000   000000000000  00000000000  000000000000   00000   00000   000")
    print(Colors.BANNERA + " 000       0000  000000000000  00000000000  000000000000  0000       0000  000")
    print('')
    print(Colors.BANNERA+'='*length + Colors.END)


def banner(string, length=STD_LENGTH):
    """Print the input `string` in a banner-like output.

    Args:
        string (str): String to be printed in banner.
        length (int): (Optional.) Number of characters used within each line.
    """
    print(Colors.BANNERA + '\n' + '='*length)
    print(Colors.BANNERA + ' '+string)
    print(Colors.BANNERA + '='*length + Colors.END)


def small_banner(string, length=STD_LENGTH):
    """Print the input `string` in a small banner-like output.

    Args:
        string (str): String to be printed in banner.
        length (int): (Optional.) Number of characters used within each line.
    """
    print(Colors.BANNERB + '\n' + '-'*length)
    print(Colors.BANNERB + ' '+string)
    print(Colors.BANNERB + '-'*length + Colors.END)


def warning(string):
    """Print the input `string` as a warning with the corresponding color.

    Args:
        string (str): String to be printed in banner.
        length (int): (Optional.) Number of characters used within each line.
    """
    print(Colors.WARN + '\n !! '+string+' !! \n'+Colors.END)


def info(string, newline=True):
    """Print the input `string` as generic output without special formatting.

    Args:
        string (str): String to be printed in banner.
        newline (bool): (Optional.) First print a newline if True.
    """
    if newline:
        print('\n# '+string)
    else:
        print('# '+string)
