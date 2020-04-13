import numpy as np
import argparse
import keras.backend as K

# Dice distance measure for non boolean values
def dice_discrete(u,v):
    c_uv = K.multiply(u, v).sum()
    c_u = K.square(u).sum()
    c_v = K.square(v).sum()
    return 1 - (2 * c_uv / (c_u + c_v + K.epsilon()))

def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(","):
             k,v = kv.split("=")
             my_dict[k] = float(v)
         setattr(namespace, self.dest, my_dict)


