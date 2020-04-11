import numpy as np
import argparse

# Dice distance measure for non boolean values
def dice_discrete(u,v):
    c_uv = np.multiply(u, v).sum()
    c_u = np.square(u).sum()
    c_v = np.square(v).sum()
    return 1 - (2 * c_uv / (c_u + c_v))

class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(","):
             k,v = kv.split("=")
             my_dict[k] = float(v)
         setattr(namespace, self.dest, my_dict)