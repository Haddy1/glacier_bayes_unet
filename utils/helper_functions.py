import numpy as np
import argparse
import subprocess as sp
from pathlib import Path
import matplotlib
import re

def set_font_size(small=8, medium=10, bigger=12):
    matplotlib.rc('font', size=small)          # controls default text sizes
    matplotlib.rc('axes', titlesize=small)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=small)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=small)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=small)    # legend fontsize
    matplotlib.rc('figure', titlesize=bigger)  # fontsize of the figure title

def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            try:
                my_dict[k] = int(v)
            except ValueError:
                try:
                    my_dict[k] = float(v)
                except ValueError:
                    my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def split_datasets(dataset):
    """
    splits combined tensorflow datasets
    :param dataset: Tensorflow dataset comprised of two or more subsets
    :return: dictionary with individiual datasets
    """
    tensors = {}
    names = list(dataset.element_spec.keys())
    for name in names:
        tensors[name] = dataset.map(lambda x: x[name])

    return tensors

def nat_sort(l):
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]

    l.sort(key=alphanum_key)



