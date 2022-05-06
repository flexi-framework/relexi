#!/usr/bin/env python3

#import sys

"""
This module contains some helper functions, to simplify and harmonize the output
to console.
"""

# Standard length of console output
STD_LENGTH=72

# Some color definitions:
class my_colors:
  END    = '\033[0m'
  RED    = '\033[91m'
  GREEN  = '\033[92m'
  YELLOW = '\033[93m'
  BLUE   = '\033[94m'


def printHeader(length=STD_LENGTH):
  """
  Print big header with program Name and Logo to console.
  TODO: Make cool ASCII art....
  """
  print(my_colors.YELLOW + '='*length)
  print(my_colors.YELLOW + '  ')
  print(my_colors.YELLOW + "  ooooooooo.             ooooo                  ooooooo  ooooo ooooo ")
  print(my_colors.YELLOW + "  `888   `Y88.           `888'                   `8888    d8'  `888' ")
  print(my_colors.YELLOW + "   888   .d88'  .ooooo.   888          .ooooo.     Y888..8P     888  ")
  print(my_colors.YELLOW + "   888ooo88P'  d88' `88b  888         d88' `88b     `8888'      888  ")
  print(my_colors.YELLOW + "   888`88b.    888ooo888  888         888ooo888    .8PY888.     888  ")
  print(my_colors.YELLOW + "   888  `88b.  888    .o  888       o 888    .o   d8'  `888b    888  ")
  print(my_colors.YELLOW + "  o888o  o888o `Y8bod8P' o888ooooood8 `Y8bod8P' o888o  o88888o o888o ")
  print(my_colors.YELLOW + '  ')
  print(my_colors.YELLOW + '='*length + my_colors.END)


def printBanner(string,length=STD_LENGTH):
  """ Print the input 'string' in a banner-like output """
  print(my_colors.YELLOW + '\n' + '='*length)
  print(my_colors.YELLOW +' '+string )
  print(my_colors.YELLOW + '='*length + my_colors.END)


def printSmallBanner(string,length=STD_LENGTH):
  """ Print the input 'string' in a banner-like output """
  print(my_colors.BLUE + '\n' + '-'*length)
  print(my_colors.BLUE +' '+string )
  print(my_colors.BLUE + '-'*length + my_colors.END)


def printWarning(string):
  """ Print the input 'string' as a warning """
  print(my_colors.RED +'\n !! '+string+' !! \n'+my_colors.END)


def printNotice(string,newline=True):
  """ Print the input 'string' as generic output without special formatting """
  if newline:
    print('\n# '+string)
  else:
    print('# '+string)
