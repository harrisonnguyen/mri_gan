import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import tensorflow as tf
from model_utils import layer_utils as layers

def generator(x,base_filter,training_ph,output_activation,n_modality,n_residuals=3):
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
        output = layers.conv_layer(output,n_modality,[7,7],1,
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

    with tf.variable_scope("final") as scope:
        output = layers.conv_layer(output,1,[3,3],1,
                            None, training_ph, None)
    return output

def cycle_gan(PATCH_SIZE,N_MODALITY,N_BASE_FILTER,N_RESIDUAL_BLOCKS,training_ph):
    x_phA = tf.placeholder(tf.float32, [None,PATCH_SIZE,PATCH_SIZE,N_MODALITY])
    x_phB = tf.placeholder(tf.float32, [None,PATCH_SIZE,PATCH_SIZE,N_MODALITY])

    with tf.variable_scope("A_to_B") as scope:
        predicted_B = generator(x_phA,N_BASE_FILTER,training_ph,
                                  output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                  n_residuals=N_RESIDUAL_BLOCKS)


    with tf.variable_scope("discrimB") as scope:
        real_B = discriminator(x_phB,N_BASE_FILTER,training_ph)
        scope.reuse_variables()
        fake_B = discriminator(predicted_B,N_BASE_FILTER,training_ph)

    with tf.variable_scope("B_to_A") as scope:
        predicted_A = generator(x_phB,N_BASE_FILTER,training_ph,
                                      output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                      n_residuals=N_RESIDUAL_BLOCKS)

    with tf.variable_scope("discrimA") as scope:
        real_A = discriminator(x_phA,N_BASE_FILTER,training_ph)
        scope.reuse_variables()
        fake_A = discriminator(predicted_A,N_BASE_FILTER,training_ph)

    with tf.variable_scope("B_to_A") as scope:
        scope.reuse_variables()
        reconstructA = generator(predicted_B,N_BASE_FILTER,training_ph,
                                       output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                       n_residuals=N_RESIDUAL_BLOCKS)

    with tf.variable_scope("A_to_B") as scope:
        scope.reuse_variables()
        reconstructB = generator(predicted_A,N_BASE_FILTER,training_ph,
                                      output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                      n_residuals=N_RESIDUAL_BLOCKS)
    return (x_phA, x_phB, predicted_A,predicted_B,
            real_A,real_B,fake_A,fake_B,reconstructA,reconstructB)
