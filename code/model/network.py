from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,LeakyReLU,UpSampling3D, Conv3D, Conv2D, UpSampling2D,Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from utils.instance_norm import InstanceNormalization

def conv2d(layer_input, filters, f_size=4):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = InstanceNormalization()(d)
    return d

def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0,
activation='relu'):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation=activation)(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = InstanceNormalization()(u)
    u = Concatenate()([u, skip_input])
    return u

def generator(input_img_shape,gf,depth):
    """U-Net Generator"""
    # Image input
    d0 = Input(shape=input_img_shape)
    input = d0
    layers = []
    # Downsampling
    for i in range(depth):
        output = conv2d(input,gf*(2**i))
        layers.append(output)
        input = output
    for i in range(depth-2, -1, -1):
        if i == 0:
            output = deconv2d(input,layers[i],gf*2**i)
        else:
            output = deconv2d(input,layers[i],gf*2**i)
        input = output

    u4 = UpSampling2D(size=2)(input)
    output_img = Conv2D(input_img_shape[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)

def d_layer(layer_input, filters, f_size=4, normalization=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if normalization:
        d = InstanceNormalization()(d)
    return d

def discriminator(input_img_shape,df,depth):
    img = Input(shape=input_img_shape)

    d1 = d_layer(img, df, normalization=False)
    input = d1
    for i in range(1,depth):
        output = d_layer(input, df*2**i)
        input = output

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(input)

    return Model(img, validity)
