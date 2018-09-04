import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import tensorflow as tf
from model_utils import layer_utils as layers
from tensorflow.contrib.layers import xavier_initializer

def residual_block3D(x, filters,training_ph,kernel_size=3, stride=1,activation=tf.nn.relu):
    """
    Creates a residual layer
    """
    padding = int((kernel_size - 1) / 2)

    # add some reflection padding
    output = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [padding, padding],[0, 0]], "REFLECT")

    with tf.variable_scope("conv1") as scope:
        output = layers.conv_3Dlayer(output,filters,kernel_size,stride,
                            activation, training_ph,'instance',padding='valid')

    output = tf.pad(output, [[0, 0], [padding, padding], [padding, padding],[padding, padding], [0, 0]], "REFLECT")

    with tf.variable_scope("conv2") as scope:
        output = layers.conv_3Dlayer(output,filters,kernel_size,stride,
                            None, training_ph,'instance',padding='valid')
    return x + output

def discriminator3D(x,base_filter,training_ph,activation=None):
    with tf.variable_scope("conv1"):
        output = layers.conv_3Dlayer(x, base_filter, 4,2,tf.nn.leaky_relu,training_ph,
                    batch_norm=None)
    with tf.variable_scope("conv2"):
            output = layers.conv_3Dlayer(output, base_filter*2, 4,2,tf.nn.leaky_relu,training_ph,
                        batch_norm='instance')
    with tf.variable_scope("conv3"):
            output = layers.conv_3Dlayer(output, base_filter*4, 4,2,tf.nn.leaky_relu,training_ph,
                        batch_norm='instance')
    with tf.variable_scope("conv4"):
            output = layers.conv_3Dlayer(output, base_filter*8, 4,2,tf.nn.leaky_relu,training_ph,
                        batch_norm='instance')
    with tf.variable_scope("final"):
            output = layers.conv_3Dlayer(output, 1, 4,1,None,training_ph,
                        batch_norm=None)
    return output

def generator3D(x,base_filter,training_ph,output_activation,n_modality,n_residuals=3):
    with tf.variable_scope("conv1") as scope:
        output = layers.conv_3Dlayer(x,base_filter,7,1,
                            tf.nn.relu, training_ph,'instance')

    with tf.variable_scope("conv2") as scope:
        output = layers.conv_3Dlayer(output,base_filter*2,3,2,
                            tf.nn.relu, training_ph,'instance')

    with tf.variable_scope("conv3") as scope:
        output = layers.conv_3Dlayer(output,base_filter*4,3,2,
                            tf.nn.relu, training_ph,'instance')

    # residual blocks
    for i in range(n_residuals):
        with tf.variable_scope("residual"+str(i)):
            output = residual_block3D(output, base_filter*4,training_ph)

    #upsample
    with tf.variable_scope("conv_transpose1") as scope:
        output = layers.conv_transpose_3Dlayer(output,base_filter*2,3,2,
                            tf.nn.relu, training_ph,'instance')


    with tf.variable_scope("conv_transpose2") as scope:
        output = layers.conv_transpose_3Dlayer(output,base_filter,3,2,
                            tf.nn.relu, training_ph,'instance')

    with tf.variable_scope("final") as scope:
        output = layers.conv_3Dlayer(output,1,7,n_modality,
                            output_activation, training_ph,None)

    return output

"""
3d_unet seems to use too much manage. cannot handle inputs of ]128,128,128]
"""
def generator3D_unet(x,base_filter,training_ph,output_activation, depth=3):
    current_output = x
    levels = []
    for i in range(depth):
        with tf.variable_scope("conv_block"+str(i)):
        # two convolutions
            with tf.variable_scope("conv1"):
                layer1 = layers.conv_3Dlayer(current_output,base_filter*(2**i),3,1,
                                    tf.nn.relu, training_ph,'instance')
            with tf.variable_scope("conv2"):
                layer2 = layers.conv_3Dlayer(layer1,base_filter*(2**i)*2,3,1,
                                    tf.nn.relu, training_ph,'instance')
            # followed by max pooling with stride 2
            if i < depth - 1:
                current_output = tf.layers.MaxPooling3D(pool_size=2,strides=2,padding='same')(layer2)
                levels.append([layer1, layer2, current_output])
            else:
                # except if we're at the end
                current_output = layer2
                levels.append([layer1, layer2])
            print(current_output.shape)
    for i in range(depth-2, -1, -1):
        with tf.variable_scope("upsample_block"+str(i)):
            with tf.variable_scope("deconv"):
                upsample = layers.conv_transpose_3Dlayer(current_output,current_output.shape[-1],
                                                    2,2,None,training_ph,None)

                concat = tf.concat([upsample,levels[i][1]],axis=-1)
            with tf.variable_scope("conv1"):
                current_output = layers.conv_3Dlayer(concat,levels[i][1].shape[1],3,1,
                                    tf.nn.relu, training_ph,'instance')
            with tf.variable_scope("conv2"):
                current_output = layers.conv_3Dlayer(current_output,levels[i][1].shape[1],3,1,
                                    tf.nn.relu, training_ph,'instance')
    with tf.variable_scope("output"):
        output = layers.conv_3Dlayer(current_output,1,1,1,
                        output_activation, training_ph,None)
    return output

def cycle_gan3D(PATCH_SIZE,N_MODALITY,N_BASE_FILTER,N_RESIDUAL_BLOCKS,training_ph):
    x_phA = tf.placeholder(tf.float32, [None,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,N_MODALITY])
    x_phB = tf.placeholder(tf.float32, [None,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,N_MODALITY])

    with tf.variable_scope("A_to_B") as scope:
        predicted_B = generator3D(x_phA,N_BASE_FILTER,training_ph,
                                  output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                  n_residuals=N_RESIDUAL_BLOCKS)


    with tf.variable_scope("discrimB") as scope:
        real_B = discriminator3D(x_phB,N_BASE_FILTER,training_ph)
        scope.reuse_variables()
        fake_B = discriminator3D(predicted_B,N_BASE_FILTER,training_ph)

    with tf.variable_scope("B_to_A") as scope:
        predicted_A = generator3D(x_phB,N_BASE_FILTER,training_ph,
                                      output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                      n_residuals=N_RESIDUAL_BLOCKS)

    with tf.variable_scope("discrimA") as scope:
        real_A = discriminator3D(x_phA,N_BASE_FILTER,training_ph)
        scope.reuse_variables()
        fake_A = discriminator3D(predicted_A,N_BASE_FILTER,training_ph)

    with tf.variable_scope("B_to_A") as scope:
        scope.reuse_variables()
        reconstructA = generator3D(predicted_B,N_BASE_FILTER,training_ph,
                                       output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                       n_residuals=N_RESIDUAL_BLOCKS)

    with tf.variable_scope("A_to_B") as scope:
        scope.reuse_variables()
        reconstructB = generator3D(predicted_A,N_BASE_FILTER,training_ph,
                                      output_activation=tf.nn.relu,n_modality=N_MODALITY,
                                      n_residuals=N_RESIDUAL_BLOCKS)
    return (x_phA, x_phB, predicted_A,predicted_B,
            real_A,real_B,fake_A,fake_B,reconstructA,reconstructB)
