import argparse
import pprint
import shutil
from pathlib import Path

import tensorflow as tf
import yaml
import numpy as np
from tensorflow import keras
from tqdm import tqdm
from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy
from models import build_model
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from layers.stn import BilinearInterpolation
import glob
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--images', type=str, required=True, help='The image file path.')
parser.add_argument('--structure', type=str, default='', help='Model Structure')
parser.add_argument('--weight', type=str, default='', required=False, help='Model Weight')
parser.add_argument('--count', type=int, default=30, required=False, help='number of image to demo')
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--pure_crnn', dest='pure_crnn', action='store_true')
feature_parser.add_argument('--no-pure_crnn', dest='pure_crnn', action='store_false')
parser.set_defaults(pure_crnn=False)
args = parser.parse_args()

if args.structure == '':
    args.structure = os.path.join(os.path.dirname(args.weight), 'structure.h5')

def read_img_and_resize(path, shape):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=shape[2])
    if shape[1] is None:
        img_shape = tf.shape(img)
        scale_factor = shape[0] / img_shape[0]
        img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
        img_width = tf.cast(img_width, tf.int32)
    else:
        img_width = shape[1]
    img = tf.image.resize(img, (shape[0], img_width))
    return img

# Specify GPU usuage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        for i in range(len(gpus)):
            mem = 1024 * 7 if i == 0 else 1024 * 9
            tf.config.set_visible_devices(gpus[i], 'GPU')
            tf.config.set_logical_device_configuration(gpus[i], [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Enable Jit to Accelerate
tf.config.optimizer.set_jit(True)

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)

batch_size = config['batch_size_per_replica']
dataset_builder = DatasetBuilder(**config['dataset_builder'])

model_old = tf.keras.models.load_model(args.structure, compile=False, custom_objects={'BilinearInterpolation': BilinearInterpolation})
if args.weight: model_old.load_weights(args.weight)

input_tensor=model_old.input
output_tensor1=model_old.get_layer('ctc_logits').output
if args.pure_crnn:
    output_tensor2=model_old.input
else:
    output_tensor2=model_old.get_layer('bilinear_interpolation').output
model = tf.keras.Model(inputs=input_tensor, outputs=[output_tensor1, output_tensor2])


model_pre = keras.layers.experimental.preprocessing.Rescaling(1./255)
model_post = CTCGreedyDecoder(config['dataset_builder']['table_path'])
model.build((None, 48, 48, 3))
model.summary()

save_model=tf.keras.Model(inputs=input_tensor, outputs=[output_tensor1])
save_model.summary()
full_crnn_run = tf.function(lambda x: save_model(x))
full_crnn_concrete_func = full_crnn_run.get_concrete_function(tf.TensorSpec([1,48,200,3], save_model.inputs[0].dtype))
save_model.save(os.path.join(os.path.dirname(args.weight), '..', 'full_crnn2'), save_format="tf", signatures=full_crnn_concrete_func)

shutil.rmtree('demo', ignore_errors=True)
os.makedirs('demo', exist_ok=True)

if os.path.isdir(args.images):
    compute_acc=False
    img_paths = []
    for prefix in ['*.jpg','*.png']:
        img_paths = img_paths + glob.glob(os.path.join(args.images, prefix))
else:
    compute_acc=True
    img_paths=[]
    label_list=[]
    label_file=open(args.images, 'r')
    label_lines=label_file.readlines()
    for line in label_lines:
        line_split=line.split()
        image_name=line_split[0]
        label_item=line_split[1]
        img_paths.append(os.path.join(os.path.dirname(args.images), image_name))
        label_list.append(label_item)

acc=0.0
total=0
for i,img_path in enumerate(img_paths):
    if i == args.count: break
    total+=1
    img_path = str(img_path)
    img = read_img_and_resize(img_path, config['dataset_builder']['img_shape'])
    img = tf.expand_dims(img, 0)
    img = tf.repeat(img, 3, axis=0)
    padding = tf.zeros((1, tf.shape(img)[1], 50, 3))

    result = model_pre(img)
    # raw_res, interpolate_img = model(result)
    # result = model_post(raw_res)
    result=save_model(result)
    print(result)
    result, prob=result[0]

    prefix=''
    if compute_acc: 
        pred=result[0].numpy()
        if pred[0].decode("utf-8")==label_list[i]: 
            acc+=1
        else:
            prefix='wrong'

    raw_res=np.argmax(raw_res, -1)
    print(f'Path: {img_path}, y_pred: {result[0].numpy()}',  f'probability: {result[1].numpy()} current acc: {acc / total}', 'raw_res: ', raw_res)

    predict_string=result[0].numpy()[0].decode('utf-8')
    embedding_string=f'{predict_string}'

    # Demonstrate Image
    img = img[0].numpy()
    img = img[..., ::-1].copy()
    img = cv2.rectangle(img, (0,0), (0 + 20, 10 ), (255,255,255), -1)
    img = cv2.putText(img, embedding_string, (0,10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    # STN Output Image
    stn_img = interpolate_img[0].numpy()
    h, w = stn_img.shape[:2]
    w = int(w * config['dataset_builder']['img_shape'][0] / h)
    h = int(config['dataset_builder']['img_shape'][0])
    stn_img = stn_img[..., ::-1]
    stn_img = cv2.resize(stn_img, (w,h))
    stn_img = (stn_img * 255.).astype(np.uint8)

    # Demo Image
    demo_img = np.hstack([img, stn_img])


    imgname = f"{prefix}_{img_path.split('/')[-1]}" if prefix != "" else img_path.split('/')[-1]
    savepath=os.path.join('demo', imgname)
    cv2.imwrite(savepath, demo_img)