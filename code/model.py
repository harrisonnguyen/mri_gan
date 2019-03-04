import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import tensorflow as tf
from model_utils import layer_utils as layers

def generator(x,base_filter,training_ph,output_activation,n_output=3,n_residuals=3):
    with tf.variable_scope("conv1") as scope:
        output = layers.conv_layer(x,base_filter,[7,7],1,
                            tf.nn.relu, training_ph,'instance')

    with tf.variable_scope("conv2") as scope:
        output = layers.conv_layer(output,base_filter*2,[3,3],2,
                            tf.nn.relu, training_ph,'instance')

    with tf.variable_scope("conv3") as scope:
        output = layers.conv_layer(output,base_filter*4,[3,3],2,
                            tf.nn.relu, training_ph,'instance')

    # residual blocks
    for i in range(n_residuals):
        with tf.variable_scope("residual"+str(i)):
            output = layers.residual_block(output, base_filter*4,training_ph)
    #upsample
    with tf.variable_scope("conv_transpose1") as scope:
        output = layers.conv_transpose_layer(output,base_filter*2,[3,3],2,
                            tf.nn.relu, training_ph,'instance')


    with tf.variable_scope("conv_transpose2") as scope:
        output = layers.conv_transpose_layer(output,base_filter,[3,3],2,
                            tf.nn.relu, training_ph,'instance')

    with tf.variable_scope("final") as scope:
        output = layers.conv_layer(output,n_output,[7,7],1,
                            output_activation, training_ph,None)
    return output

def discriminator(x,base_filter,training_ph):
    with tf.variable_scope("conv1") as scope:
        output = layers.conv_layer(x,base_filter,[4,4],1,
                            tf.nn.leaky_relu, training_ph,None)

    with tf.variable_scope("conv2") as scope:
        output = layers.conv_layer(output,base_filter*2,[4,4],2,
                            tf.nn.leaky_relu, training_ph,'instance')

    with tf.variable_scope("conv3") as scope:
        output = layers.conv_layer(output,base_filter*4,[4,4],2,
                            tf.nn.leaky_relu, training_ph,'instance')

    with tf.variable_scope("conv4") as scope:
        output = layers.conv_layer(output,base_filter*8,[4,4],1,
                            tf.nn.leaky_relu, training_ph,'instance')
        feature_map = output
    kernel_size = x.get_shape()[1]/16

    # produce a 1x1x1 output
    with tf.variable_scope("final") as scope:
        output = layers.conv_layer(output,1,[kernel_size,kernel_size],1,
                            None, training_ph, None)
    return output,feature_map
