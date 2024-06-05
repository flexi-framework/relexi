#!/usr/bin/env python3

"""Contains helper functions to harmonize the output to console."""

STD_LENGTH = 72
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

    TODO:
        Make cool ASCII art.
    """
    print(Colors.BANNERA+'='*length)
    print(Colors.BANNERA+'  ')
    print(Colors.BANNERA +
        " ooooooooo.             ooooo                  ooooooo  ooooo ooooo")
    print(Colors.BANNERA +
        " `888   `Y88.           `888'                   `8888    d8'  `888'")
    print(Colors.BANNERA +
        "  888   .d88'  .ooooo.   888          .ooooo.     Y888..8P     888 ")
    print(Colors.BANNERA +
        "  888ooo88P'  d88' `88b  888         d88' `88b     `8888'      888 ")
    print(Colors.BANNERA +
        "  888`88b.    888ooo888  888         888ooo888    .8PY888.     888 ")
    print(Colors.BANNERA +
        "  888  `88b.  888    .o  888       o 888    .o   d8'  `888b    888 ")
    print(Colors.BANNERA +
        " o888o  o888o `Y8bod8P' o888ooooood8 `Y8bod8P' o888o  o88888o o888o")
    print(Colors.BANNERA+'  ')
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
