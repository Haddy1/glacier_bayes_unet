from keras import backend as K
import keras.losses
import tensorflow as tf
from scipy.spatial.distance import dice

def focal_loss_nieradzik(alpha=1, gamma=0):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss

def focal_loss(alpha=1, gamma=0):
    def loss(y_true, y_pred):
        y_pred_stable = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        weight_tp = alpha * (1 - y_pred) ** gamma * y_true * K.log(y_pred_stable)
        weight_tn = alpha * y_pred ** gamma * (1 - y_true) * K.log(1 - y_pred_stable)

        return -K.mean(weight_tp + weight_tn)

    return loss


def dice_loss(y_true, y_pred):
#    numerator = 2 * K.sum(y_true * y_pred, axis=-1)
#    denominator = K.sum(y_true + y_pred, axis=-1)
#
#    return 1 - (numerator + 1) / (denominator + 1)
    return dice(y_true, y_pred)


def combined_loss(loss_functions, split):

    def loss(y_true, y_pred):

        combined_result = 0
        for loss_function, weight in zip(loss_functions,split):
            combined_result += weight * loss_function(y_true, y_pred)

        return combined_result

    return loss