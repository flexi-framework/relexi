#!/usr/bin/env python3

"""
This module contains some helper functions, to simplify and harmonize the output
to console.
"""

STD_LENGTH=72
"""Standard length for output to console."""


class colors:
  """Print big header with program name and logo to console.

  Attributes:
    BANNERA (str): Defines first banner color for Header and large banners.
    BANNERB (str): Defines second banner color for small banners.
    WARN    (str): Defines color for warnings.
    END     (str): Defines end of color in string.
  """
  BANNERA = '\033[93m'
  BANNERB = '\033[94m'
  WARN    = '\033[91m'
  END     = '\033[0m'


def printHeader(length=STD_LENGTH):
  """Print big header with program name and logo to console.

  Args:
    length (int): Number of characters used within each line.

  Returns:
    None

  TODO:
    Make cool ASCII art.
  """
  print(colors.BANNERA + '='*length)
  print(colors.BANNERA + '  ')
  print(colors.BANNERA + "  ooooooooo.             ooooo                  ooooooo  ooooo ooooo ")
  print(colors.BANNERA + "  `888   `Y88.           `888'                   `8888    d8'  `888' ")
  print(colors.BANNERA + "   888   .d88'  .ooooo.   888          .ooooo.     Y888..8P     888  ")
  print(colors.BANNERA + "   888ooo88P'  d88' `88b  888         d88' `88b     `8888'      888  ")
  print(colors.BANNERA + "   888`88b.    888ooo888  888         888ooo888    .8PY888.     888  ")
  print(colors.BANNERA + "   888  `88b.  888    .o  888       o 888    .o   d8'  `888b    888  ")
  print(colors.BANNERA + "  o888o  o888o `Y8bod8P' o888ooooood8 `Y8bod8P' o888o  o88888o o888o ")
  print(colors.BANNERA + '  ')
  print(colors.BANNERA + '='*length + colors.END)


def printBanner(string,length=STD_LENGTH):
  """Print the input `string` in a banner-like output.

  Args:
    string (str): String to be printed in banner.
    length (int): (Optional.) Number of characters used within each line.

  Returns:
    None
  """
  print(colors.BANNERA + '\n' + '='*length)
  print(colors.BANNERA +' '+string )
  print(colors.BANNERA + '='*length + colors.END)


def printSmallBanner(string,length=STD_LENGTH):
  """Print the input `string` in a small banner-like output.

  Args:
    string (str): String to be printed in banner.
    length (int): (Optional.) Number of characters used within each line.

  Returns:
    None
  """
  print(colors.BANNERB + '\n' + '-'*length)
  print(colors.BANNERB +' '+string )
  print(colors.BANNERB + '-'*length + colors.END)


def printWarning(string):
  """Print the input `string` as a warning with the corresponding color.

  Args:
    string (str): String to be printed in banner.
    length (int): (Optional.) Number of characters used within each line.

  Returns:
    None
  """
  print(colors.WARN +'\n !! '+string+' !! \n'+colors.END)


def printNotice(string,newline=True):
  """Print the input `string` as generic output without special formatting.

  Args:
    string (str): String to be printed in banner.
    newline (bool): (Optional.) First print a newline if True.

  Returns:
    None
  """
  if newline:
    print('\n# '+string)
  else:
    print('# '+string)
