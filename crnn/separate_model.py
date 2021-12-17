import argparse
from pathlib import Path
import cv2
import yaml
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from dataset_factory import DatasetBuilder
from losses import CTCLoss, LossBox
from metrics import SequenceAccuracy, EditDistance
from models import build_model
from layers.stn import BilinearInterpolation
print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--stn_weight', type=str, required=False, help='The weight path to stn_model(h5 file)')
parser.add_argument('--crnn_weight', type=str, required=False, help='The weight path to crnn_model(h5 file)')
parser.add_argument('--merge', dest='merge', default=False, action='store_true', help="Merge heavy stn and crnn model.")
parser.add_argument('--merge_path', type=str, required=False, help="The path to save merge model")
parser.add_argument('--config', type=Path, required=True, help="""
The training configuration to the target mdoel
If merge == true, then it should be the config file of stn_weight.
""")
args = parser.parse_args()
with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']

def test_model(model):
    # Testing Load and Inference pb graph
    print(list(model.signatures.keys()))
    infer = model.signatures[list(model.signatures.keys())[0]]
    print('infer.structured_input_signature', infer.structured_input_signature)
    print('infer.structured_outputs', infer.structured_outputs)
    INPUT_SHAPE=tuple(infer.structured_input_signature[1]['x'].shape)
    # pb Inference
    image = np.zeros(INPUT_SHAPE).astype(np.float32)
    print('input image detaial', image.shape, image.dtype)
    result = model(image)
    print("successfully do inference in savemodel")
    print('result of output', result.shape, result.dtype)

if args.stn_weight is not None:
    low_resolution_shape=config['dataset_builder']['img_shape']
    low_resolution_shape.insert(0, 1)
    ###############################################
    ##########     Loading Model File    ##########
    ###############################################
    model = tf.keras.models.load_model(
        os.path.join(os.path.dirname(args.stn_weight), 'structure.h5'),
        compile=False, custom_objects={ 'BilinearInterpolation': BilinearInterpolation }
    )
    model.load_weights(args.stn_weight)
    model.summary()
    
    ###############################################
    #####   Processing Heavy STN Model   ##########
    ###############################################
    # Low resolution h,w
    height, width = low_resolution_shape[1:3]
    apsect_w_h = width / height
    # High resolution h,w
    high_resolution_shape=(1, 64, int(64*apsect_w_h), 3)

    input=tf.keras.Input(high_resolution_shape[1:], name='heavy_stn_input')
    # Resize to low resolution, so that model can compute faster
    x0 = tf.image.resize(input, low_resolution_shape[1:3])
    stn_mat=model(x0)
    x0=BilinearInterpolation((48, 48))([input, stn_mat])

    ###############################################
    #########   Construct STN Model   #############
    ###############################################
    heavy_stn =tf.keras.Model(input, x0, name='heavy_stn')
    heavy_stn.compile()
    if not args.merge:
        heavy_stn_model_path=os.path.join(os.path.dirname(args.stn_weight), '..', 'heavy_stn')
        heavy_stn_run = tf.function(lambda x: heavy_stn(x))
        heavy_stn_concrete_func = heavy_stn_run.get_concrete_function(tf.TensorSpec(high_resolution_shape, heavy_stn.inputs[0].dtype))
        heavy_stn.save(heavy_stn_model_path, save_format="tf", signatures=heavy_stn_concrete_func)
        print('Test Loading Model: ', heavy_stn_model_path)
        heavy_stn=tf.keras.models.load_model(heavy_stn_model_path)
        test_model(heavy_stn)

if args.crnn_weight is not None:
    ###############################################
    ##########     Loading Model File    ##########
    ###############################################
    model = tf.keras.models.load_model(
        os.path.join(os.path.dirname(args.crnn_weight), 'structure.h5'),
        compile=False, custom_objects={'BilinearInterpolation': BilinearInterpolation }
    )
    model.load_weights(args.crnn_weight)
    model.summary()
    model_input_shape=model.input.shape
    image_shape=(1, *model_input_shape[1:])

    ###############################################
    ##########     Construct CRNN MOdel    ########
    ###############################################
    # input_tensor=model.input
    input_tensor = model.input
    print(input_tensor)
    x0=model.get_layer('ctc_logits').output
    # x0=CTCGreedyDecoder(config['dataset_build er']['table_path'])(x0)
    crnn_model=tf.keras.Model(input_tensor, x0, name='crnn_model')
    crnn_model.compile()
    crnn_model.summary()
    if not args.merge:
        crnn_model_path=os.path.join(os.path.dirname(args.crnn_weight), '..', 'crnn_model')
        crnn_model_run = tf.function(lambda x: crnn_model(x))
        crnn_model_concrete_func = crnn_model_run.get_concrete_function(tf.TensorSpec(image_shape, crnn_model.inputs[0].dtype))
        crnn_model.save(crnn_model_path, save_format="tf", signatures=crnn_model_concrete_func)
        print('Test Loading Model: ', crnn_model_path)
        crnn_model=tf.keras.models.load_model(crnn_model_path)
        test_model(crnn_model)

if args.merge:
    merge_model_path=args.merge_path if args.merge_path is not None else './default_export'
    input=heavy_stn.input
    x=crnn_model(heavy_stn.output)
    merge_model=tf.keras.Model(input, x, name='merge_model')
    merge_model.compile(
        
    )
    merge_model_run = tf.function(lambda x: merge_model(x))
    merge_model_concrete_func = merge_model_run.get_concrete_function(tf.TensorSpec(high_resolution_shape, heavy_stn.inputs[0].dtype))
    merge_model.save(merge_model_path, save_format="tf", signatures=merge_model_concrete_func)
    print('Test Loading Model: ', merge_model_path)
    merge_model=tf.keras.models.load_model(merge_model_path)
    test_model(merge_model)
