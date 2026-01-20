import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
def asymmetric_huber(
    y_true,
    y_pred,
    delta=0.5,
    under_weight=2.2,
    over_weight=1.2
):
    error = y_pred - y_true
    abs_error = tf.abs(error)

    huber = tf.where(
        abs_error <= delta,
        0.5 * tf.square(error),
        delta * (abs_error - 0.5 * delta)
    )

    weight = tf.where(error < 0, under_weight, over_weight)
    return tf.reduce_mean(weight * huber)