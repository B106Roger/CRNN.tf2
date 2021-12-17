import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
from layers.stn import BilinearInterpolation
from tensorflow import keras
from tensorflow.keras import layers
reg=1e-3

def separable_conv(x, p_filters, d_kernel_size=(3,3), d_strides=(1,1), d_padding='valid', reg=None):
    x = layers.DepthwiseConv2D(kernel_size=d_kernel_size, strides=d_strides, padding=d_padding, use_bias=True, kernel_regularizer=reg)(x)
    x = layers.Conv2D(p_filters, kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    return x

def get_initial_weights(output_size):
    b = np.random.normal(0.0, 0.001, (2, 3))            # init weight zero won't trigger backpropagation
    b[0, 0] = 0.8 #0.25
    b[1, 1] = 0.8 #0.5
    W = np.random.normal(0.0, 0.01, (output_size, 6))  # init weight zero won't trigger backpropagation
    weights = [W, b.flatten()]
    return weights

def vgg_style(x, reg=None):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
    """
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=reg)(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 2), padding='same')(x)

    x = separable_conv(x, p_filters=512, d_kernel_size=(3,3), d_strides=(1,1), d_padding='same', reg=reg)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)
    x = separable_conv(x, p_filters=512, d_kernel_size=(3,3), d_strides=(1,1), d_padding='valid', reg=reg)

    x = layers.Reshape((-1, 512))(x)
    return x


##############################################
##############################################
##############################################
##############################################

def build_stn(img, interpolation_size, slice=False, reg=None):
    x = layers.Conv2D(32, (5, 5), padding='SAME', use_bias=False, kernel_regularizer=reg)(img) # 20
    x = layers.BatchNormalization()(x)
    x0 = layers.ReLU(6)(x)
    
    x = x0
    if slice:
        x = layers.Conv2D(64, (5, 2), padding='SAME', use_bias=False, kernel_regularizer=reg)(x0)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6)(x)
        s_logits = layers.Conv2D(1, (6, 1), padding='valid', dilation_rate=(2,1), use_bias=True, kernel_regularizer=reg)(x)
        s_logits = layers.Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True), name='slice_logits')(s_logits)
        s_sigmoid = layers.Activation(tf.math.sigmoid, name='slice_sigmoid')(s_logits)
        x = layers.Multiply(name='slice_result')([x0, s_sigmoid])

    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (5, 5), padding='SAME', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='SAME', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)

    x1 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=1, use_bias=False, kernel_regularizer=reg)(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU(6)(x1)

    x2 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=2, use_bias=False, kernel_regularizer=reg)(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU(6)(x2)

    x3 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=3, use_bias=False, kernel_regularizer=reg)(x)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU(6)(x3)

    x = layers.Concatenate()([x1,x2,x3])
    x = layers.Conv2D(256, (1, 1), padding='SAME', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x) #10x50
    # x = layers.ReLU(6)(x)
    # TODO change to global max pooling
    # TODO increasing channel number
    x = tfa.layers.SpatialPyramidPooling2D([[6,9],[4,6],[2,3]])(x) # 17408

    x = layers.Flatten()(x)
    x = layers.Dense(32, use_bias=False, kernel_regularizer=reg)(x) # 32
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    transform_mat = layers.Dense(6, weights=get_initial_weights(32), name="stn")(x)
    interpolated_image = BilinearInterpolation(interpolation_size, name='bilinear_interpolation')([img, transform_mat])
    if slice:
        return interpolated_image, transform_mat, s_logits
    else:
        return interpolated_image, transform_mat, None

def build_stn2(img, interpolation_size, slice=False):
    h = img.shape[1]
    w = img.shape[2]
    reg=1e-3
    x = layers.Conv2D(64, (5, 5), padding='SAME', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(img) # 20
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(128, (5, 5), padding='SAME', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding='SAME', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    
    x1 = layers.Conv2D(256, (3, 3), padding='SAME', dilation_rate=1, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU(6)(x1)

    x2 = layers.Conv2D(256, (3, 3), padding='SAME', dilation_rate=2, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU(6)(x2)

    x3 = layers.Conv2D(256, (3, 3), padding='SAME', dilation_rate=3, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU(6)(x3)

    x = layers.Concatenate()([x1,x2,x3])
    x = layers.Conv2D(512, (1, 1), padding='SAME', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x = layers.BatchNormalization()(x) #10x50
    # x = layers.ReLU(6)(x)
    # TODO change to global max pooling
    # TODO increasing channel number
    if h == 16:
        x = tfa.layers.SpatialPyramidPooling2D([[4,10],[4,5],[2,4]])(x) # (40+20+8)*512=68*512=34816
    else:
        x = tfa.layers.SpatialPyramidPooling2D([[6,9],[4,6],[2,3]])(x) # (54+24+6)*512=84*512=43008

    x = layers.Flatten()(x)
    x = layers.Dense(64, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x) # 32
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    transform_mat = layers.Dense(6, weights=get_initial_weights(64), name="stn")(x)
    interpolated_image = BilinearInterpolation(interpolation_size, name='bilinear_interpolation')([img, transform_mat])
    if slice:
        return interpolated_image, transform_mat, None
    else:
        return interpolated_image, transform_mat, None

###############################################
### Build Model for light STN and CRNN     ####
###############################################
def build_model(num_classes,
                use_stn=False,
                train_stn=False,
                slice=False,
                weight=None,
                img_shape=(32, None, 3),
                model_name='crnn'):
    x = img_input = keras.Input(shape=img_shape)
    if use_stn: interpolate_img, transform_mat, s_logits = build_stn(x, (48, 48), slice, reg=tf.keras.regularizers.L2(reg))
    else: interpolate_img = x

    x = vgg_style(interpolate_img, reg=tf.keras.regularizers.L2(reg))
    x = layers.Reshape((1, 4, 512))(x)
    x = layers.Conv2D(512, (1,4), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(1e-2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.Conv2D(512, (1,4), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(1e-2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.Reshape((-1, 512))(x)
    x = layers.Dense(units=num_classes, name='ctc_logits')(x)
    
    if use_stn:
        if train_stn:
            if slice:
                model = keras.Model(inputs=img_input, outputs=[x, transform_mat, s_logits], name=model_name)
            else:
                model = keras.Model(inputs=img_input, outputs=[x, transform_mat], name=model_name)
        else:
            model = keras.Model(inputs=img_input, outputs=x, name=model_name)

        model_vis = keras.Model(inputs=img_input, outputs=[interpolate_img, transform_mat])
    else:
        model = keras.Model(inputs=img_input, outputs=x, name=model_name)
        model_vis=keras.Model(inputs=img_input, outputs=[x, img_input], name=model_name)
    
    if weight: model.load_weights(weight)
    return model, model_vis

###############################################
### Build Model for Heavy STN              ####
###############################################
def build_pure_stn(img_shape, interpolation_size, model_type):
    input=tf.keras.Input(shape=img_shape)
    if model_type == 1:
        interpolated_image, transform_mat, _=build_stn(input, interpolation_size)
    elif model_type == 2:
        interpolated_image, transform_mat, _=build_stn2(input, interpolation_size)

    train_model=tf.keras.Model(input, transform_mat)
    visualize_model=tf.keras.Model(input, [interpolated_image, transform_mat])
    return train_model, visualize_model

###############################################
### Build Model for CRNN                   ####
###############################################
# def build_pure_model(num_classes, weight=None, img_shape=(32, None, 3), use_stn=False, model_name='crnn'):
#     x = img_input = keras.Input(shape=img_shape)
#     if use_stn:
#         interpolate_img, transform_mat, _ = build_stn(x, (48, 48), slice)
#     else:
#         interpolate_img=x
#     x = vgg_style(img_input, reg=tf.keras.regularizers.L2(reg))

#     x = layers.Reshape((1, 4, 512))(x)
#     x = layers.Conv2D(512, (1,4), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(6)(x)
#     x = layers.Conv2D(512, (1,4), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(6)(x)
#     x = layers.Reshape((-1, 512))(x)
#     x = layers.Dense(units=num_classes, name='ctc_logits')(x)

#     model = keras.Model(inputs=img_input, outputs=[x, transform_mat], name=model_name)
#     model_vis = keras.Model(inputs=img_input, outputs=[x, interpolate_img])
#     if weight:
#         model.load_weights(weight, by_name=True, skip_mismatch=True)
        
#     return model, model_vis