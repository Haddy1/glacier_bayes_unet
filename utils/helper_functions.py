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
    c_uv = np.sum(u*v)
    if c_uv == 0:
        return 0
    else:
        c_u = np.sum(u**2)
        c_v = np.sum(v**2)
    return 2 * c_uv / (c_u + c_v)

def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = np.sum(neg_y_true * y_pred)
    tn = np.sum(neg_y_true * neg_y_pred)
    result = tn / (tn + fp + K.epsilon())
    return result

def IOU(y_true, y_pred):
    intersection = np.sum(y_true == y_pred)
    union = y_true.size
    return intersection / union


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


