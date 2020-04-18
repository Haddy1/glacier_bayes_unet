import numpy as np
import argparse
import keras.backend as K

def dice_coefficient(u,v):
    """
    For binary vectors the Dice cooefficient can be written as
    2 * |u * v| / (|u**2| + |v**2|)

    | u * v | gives intersecting set
    |u**2|, |v**2| number of (true) elements in set

    :param u:  binary vector
    :param v:  binary vector of same length as u
    :return:   dice coefficient
    """
    c_uv = K.sum(u*v)
    c_u = K.sum(u**2)
    c_v = K.sum(v**2)
    return 2 * K.sum(u * v)/ (K.sum(u**2) + K.sum(v**2) + K.epsilon())

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


