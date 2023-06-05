#!/usr/bin/env python3

from .context import relexi
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
