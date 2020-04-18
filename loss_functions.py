from keras import backend as K
import keras.losses
import tensorflow as tf
from scipy.spatial.distance import dice

def focal_loss(alpha=1, gamma=1.5):
    def loss(y_true, y_pred):
        """
        Focal loss using the binary crossentropy implementation of keras as blueprint
        :param y_true:  True labels
        :param y_pred:  Predictions of the the same shape as y_true
        :return: loss value
        """
        epsilon = K.epsilon()
        y_pred_stable = K.clip(y_pred, epsilon, 1 - epsilon)
        weight_tp = (1 - y_pred_stable) ** gamma * y_true * K.log(y_pred_stable + epsilon)
        weight_tn = y_pred_stable ** gamma * (1 - y_true) * K.log(1 - y_pred_stable + epsilon)

        return - alpha * K.mean(weight_tp + weight_tn)

    return loss




def weighted_dice_loss(beta):
    def dice_loss(y_true, y_pred):
        numerator = 2 * K.sum(y_true * y_pred, axis=-1)
        denominator = K.sum(y_true + y_pred, axis=-1)

        return 1 - (numerator + 1) / (denominator + 1)


def combined_loss(loss_functions, split):

    def loss(y_true, y_pred):

        combined_result = 0
        for loss_function, weight in zip(loss_functions,split):
            combined_result += weight * loss_function(y_true, y_pred)

        return combined_result

    return loss