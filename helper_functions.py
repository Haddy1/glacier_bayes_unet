import numpy as np

# Dice distance measure for non boolean values
def dice_discrete(u,v):
    c_uv = np.multiply(u, v).sum()
    c_u = np.square(u).sum()
    c_v = np.square(v).sum()
    return 1 - (2 * c_uv / (c_u + c_v))
