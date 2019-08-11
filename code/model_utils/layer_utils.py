import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d,xavier_initializer

def conv_layer(x,filters,kernel_size,strides,activation,training_ph,
                batch_norm='batch',padding='same'):
    output = tf.layers.conv2d(
                    inputs=x,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation=None,
                    kernel_initializer=xavier_initializer_conv2d(),
                )
    if batch_norm == 'batch':
        output = tf.layers.batch_normalization(
                    inputs=output,
                    training=training_ph)
    elif  batch_norm == 'instance':
        output = tf.contrib.layers.instance_norm(
                inputs=output)

    if activation is not None:
        output = activation(output)
    return output

def conv_transpose_layer(x,filters,kernel_size,strides,
                         activation,training_ph,batch_norm='batch',padding='same'):
    output = tf.layers.conv2d_transpose(
                    inputs=x,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    activation=None,
                    kernel_initializer=xavier_initializer_conv2d(),
                )
    if batch_norm == 'batch':
        output = tf.layers.batch_normalization(
                    inputs=output,
                    training=training_ph)
    elif  batch_norm == 'instance':
        output = tf.contrib.layers.instance_norm(
                inputs=output)
    if activation is not None:
        output = activation(output)
    return output


def residual_block(x, filters,training_ph,kernel_size=3, stride=1,activation=tf.nn.relu):
    """
    Creates a residual layer
    """
    padding = int((kernel_size - 1) / 2)

    # add some reflection padding
    output = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")

    with tf.variable_scope("conv1") as scope:
        output = conv_layer(output,filters,[kernel_size,kernel_size],stride,
                            activation, training_ph,'instance',padding='valid')

    output = tf.pad(output, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "REFLECT")

    with tf.variable_scope("conv2") as scope:
        output = conv_layer(output,filters,[kernel_size,kernel_size],stride,
                            None, training_ph,'instance',padding='valid')
    return x + output

def conv_3Dlayer(x,filters,kernel_size,strides,activation,training_ph,batch_norm=None,padding='same'):
    output = tf.layers.conv3d(
                inputs=x,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=None,
                kernel_initializer=xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
            )
    if batch_norm == 'batch':
        output = tf.layers.batch_normalization(
                    inputs=output,
                    training=training_ph)
    elif  batch_norm == 'instance':
        output = tf.contrib.layers.instance_norm(
                inputs=output)
    if activation is not None:
        output = activation(output)
    return output

def conv_transpose_3Dlayer(x,filters,kernel_size,strides,
                         activation,training_ph,batch_norm='batch',padding='same'):
    output = tf.layers.conv3d_transpose(
                    inputs=x,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation=None,
                    kernel_initializer=xavier_initializer(),
                )
    if batch_norm == 'batch':
        output = tf.layers.batch_normalization(
                    inputs=output,
                    training=training_ph)
    elif  batch_norm == 'instance':
        output = tf.contrib.layers.instance_norm(
                inputs=output)
    if activation is not None:
        output = activation(output)
    return output
