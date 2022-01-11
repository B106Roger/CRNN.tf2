import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import os
from layers.stn import BilinearInterpolation
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
reg=1e-2

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope
LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer
tfmot.quantization.keras.QuantizeConfig

class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """QuantizeConfig which does not quantize any part of the layer."""
    def get_weights_and_quantizers(self, layer):
        return []
    def get_activations_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_activations):
        pass
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}

def apply_quantization_to_all(layer):
    print(layer)
    if isinstance(layer, tfa.layers.SpatialPyramidPooling2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, NoOpQuantizeConfig())
    elif isinstance(layer, BilinearInterpolation):
        return layer
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        return layer
    elif isinstance(layer, tf.keras.layers.Concatenate):
        return layer
    else:
        return tfmot.quantization.keras.quantize_annotate_layer(layer)


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

    return x


##############################################
##############################################
##############################################
##############################################

def build_stn(img, interpolation_size, slice=False, reg=None, qat=False):
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
    # x1 = layers.ReLU(6)(x1)

    x2 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=2, use_bias=False, kernel_regularizer=reg)(x)
    x2 = layers.BatchNormalization()(x2)
    # x2 = layers.ReLU(6)(x2)

    x3 = layers.Conv2D(128, (3, 3), padding='SAME', dilation_rate=3, use_bias=False, kernel_regularizer=reg)(x)
    x3 = layers.BatchNormalization()(x3)
    # x3 = layers.ReLU(6)(x3)

    x = layers.Concatenate()([x1,x2,x3])
    x = layers.ReLU(6)(x)

    x = layers.Conv2D(256, (1, 1), padding='SAME', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x) #10x50
    # x = layers.ReLU(6)(x)
    # TODO change to global max pooling
    # TODO increasing channel number
    x = tfa.layers.SpatialPyramidPooling2D([[6,9],[4,6],[2,3]])(x) # 17408

    x = layers.Reshape((1,1,-1))(x)
    x = layers.Conv2D(32, (1, 1), padding='SAME', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)

    # x = layers.Flatten()(x)
    # x = layers.Dense(32, use_bias=False, kernel_regularizer=reg)(x) # 32
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU(6)(x)
    transform_mat = layers.Dense(6, weights=get_initial_weights(32), name="stn")(x)
    interpolated_image = BilinearInterpolation(interpolation_size, name='bilinear_interpolation')([img, transform_mat])
    if slice:
        return interpolated_image, transform_mat, s_logits
    else:
        return interpolated_image, transform_mat, None

def build_stn2(img, interpolation_size, slice=False, qat=False):
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
    # x1 = layers.ReLU(6)(x1)

    x2 = layers.Conv2D(256, (3, 3), padding='SAME', dilation_rate=2, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x2 = layers.BatchNormalization()(x2)
    # x2 = layers.ReLU(6)(x2)

    x3 = layers.Conv2D(256, (3, 3), padding='SAME', dilation_rate=3, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x3 = layers.BatchNormalization()(x3)
    # x3 = layers.ReLU(6)(x3)

    x = layers.Concatenate()([x1,x2,x3])
    x = layers.ReLU(6)(x)

    x = layers.Conv2D(512, (1, 1), padding='SAME', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x = layers.BatchNormalization()(x) #10x50
    # x = layers.ReLU(6)(x)
    # TODO change to global max pooling
    # TODO increasing channel number
    if h == 16:
        # x = quantize_annotate_layer(tfa.layers.SpatialPyramidPooling2D([[4,10],[4,5],[2,4]]), NoOpQuantizeConfig())(x)
        x = tfa.layers.SpatialPyramidPooling2D([[4,10],[4,5],[2,4]])(x) # (40+20+8)*512=68*512=34816
        neurons=68
    else:
        # x = quantize_annotate_layer(tfa.layers.SpatialPyramidPooling2D([[6,9],[4,6],[2,3]]), NoOpQuantizeConfig())(x)
        x = tfa.layers.SpatialPyramidPooling2D([[6,9],[4,6],[2,3]])(x) # (54+24+6)*512=84*512=43008
        neurons=84
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(reg))(x) # 32
    
    x = layers.Reshape((1,1,-1))(x)
    x = layers.Conv2D(64, (1, 1), padding='SAME', use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6)(x)
    x = layers.Reshape((-1,))(x)
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

        model_vis = keras.Model(inputs=img_input, outputs=[x, interpolate_img])
    else:
        model = keras.Model(inputs=img_input, outputs=x, name=model_name)
        model_vis=keras.Model(inputs=img_input, outputs=[x, img_input], name=model_name)
    
    if weight: model.load_weights(weight)
    return model, model_vis

###############################################
### Build Model for Heavy STN              ####
###############################################
def build_pure_stn(img_shape, interpolation_size, model_type, qat=False):
    input=tf.keras.Input(shape=img_shape)
    if model_type == 1:
        interpolated_image, transform_mat, _=build_stn(input, interpolation_size, qat)
    elif model_type == 2:
        interpolated_image, transform_mat, _=build_stn2(input, interpolation_size, qat)

    train_model=tf.keras.Model(input, transform_mat)
    visualize_model=tf.keras.Model(input, [interpolated_image, transform_mat])
    return train_model, visualize_model


class InferenceModel:
    def __init__(self, model_path, model_name, model_type, is_qat_model=False):
        """
        model_path: string, path of model
        model_name: string, name of model
        model_type: string, type of model(heavy_stn, crnn_model)
        """

        ###################################
        ##### Load Model Weight     #######
        ###################################
        model_format=None
        if os.path.isfile(model_path) and '.h5' in model_path:
            self.model_format='hdf5'
            if is_qat_model:
                with tfmot.quantization.keras.quantize_scope({
                    'NoOpQuantizeConfig': NoOpQuantizeConfig,
                    'BilinearInterpolation':BilinearInterpolation,
                }):
                    model = tf.keras.models.load_model(os.path.join(os.path.dirname(model_path), 'structure.h5'), custom_objects={
                        'BilinearInterpolation': BilinearInterpolation,
                        'SpatialPyramidPooling2D': tfa.layers.SpatialPyramidPooling2D,
                    }, compile=False)
                    model.load_weights(model_path)
            else:
                model = tf.keras.models.load_model(os.path.join(os.path.dirname(model_path), 'structure.h5'), custom_objects={
                    'BilinearInterpolation': BilinearInterpolation,
                    'SpatialPyramidPooling2D': tfa.layers.SpatialPyramidPooling2D,
                }, compile=False)
                model.load_weights(model_path)
        elif os.path.isdir(model_path):
            self.model_format='savemodel'
            model = tf.keras.models.load_model(model_path, custom_objects={
                'BilinearInterpolation': BilinearInterpolation,
                'SpatialPyramidPooling2D': tfa.layers.SpatialPyramidPooling2D,
            }, compile=False)
        elif os.path.isfile(model_path) and '.tflite' in model_path:
            self.model_format='tflite'

            interpreter = tf.lite.Interpreter(model_path, num_threads=16)
            interpreter.allocate_tensors()
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()

            model=interpreter
        else:
            raise ValueError(f'Unknown model file {os.path.basename(model_path)}')
        
        ########################################
        #######  Reconstruct Model Output   ####
        ########################################
        if model_type=='heavy_stn':
            if self.model_format=='savemodel':
                model = tf.keras.models.Model(model.input, [
                    model.get_layer('bilinear_interpolation').output,
                    model.get_layer('functional_1').get_layer('stn').output
                ])
            elif self.model_format=='hdf5':
                # aspect_ratio=w/h
                model_input_aspect_ratio=model.input.shape[2] / model.input.shape[1]
                high_resolution_shape=(64, int(64*model_input_aspect_ratio), 3)
                low_resolution_shape=model.input.shape[1:]

                model_input=tf.keras.Input(high_resolution_shape)
                x = tf.image.resize(model_input, low_resolution_shape[:2])
                stn_mat = model(x)
                stn_image=BilinearInterpolation((48,48))([model_input, stn_mat])
                model = tf.keras.Model([model_input], [stn_image, stn_mat])

        elif model_type=='crnn_model':
            if self.model_format!='tflite':
                if is_qat_model:
                    model = tf.keras.Model(model.input, [
                        model.get_layer('quant_ctc_logits').output
                    ])
                else:
                    model = tf.keras.Model(model.input, [
                        model.get_layer('ctc_logits').output
                    ])
            

        if self.model_format!='tflite': model.compile()

        self.model = model
    
    def __call__(self, input_tensor):
        if self.model_format=='hdf5':
            return self.model(input_tensor)
        elif self.model_format=='savemodel':
            return self.model(input_tensor)
        elif self.model_format=='tflite':
            # Inference Method 1
            self.model.set_tensor(self.input_details[0]["index"], input_tensor)
            self.model.invoke()
            # output=[]
            # for i in range(len(self.output_details)):
            #     output.append(self.model.get_tensor(self.output_details[i]["index"]))
            output=[self.model.get_tensor(self.output_details[i]["index"]) for i in range(len(self.output_details))]
            if len(output) == 1:
                return output[0]
            else:
                return output

    @property
    def input_shape(self):
        if self.model_format != 'tflite':
            return self.model.input.shape
        elif self.model_format == 'tflite':
            return self.input_details[0]['shape']