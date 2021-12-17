import argparse
from pathlib import Path
import cv2
import yaml
import tensorflow as tf
import os
import numpy as np
from models import build_model
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from layers.stn import BilinearInterpolation

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--weight', type=str, required=True, default='', help='The saved weight path.')
parser.add_argument('--pre', type=str, help='pre processing.')
parser.add_argument('--post', type=str, help='Post processing.')
args = parser.parse_args()
args.structure=os.path.join(os.path.dirname(args.weight), 'structure.h5')
args.output_dir='export_dir'
os.makedirs(args.output_dir, exist_ok=True)
for subdir in ['pb_dir', 'tflite_dir']:
    os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)


with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']

with open(config['table_path']) as f:
    num_classes = len(f.readlines())

if args.pre == 'rescale':
    preprocess = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
else:
    preprocess = None

if args.post == 'softmax':
    postprocess = tf.keras.layers.Softmax()
elif args.post == 'greedy':
    postprocess = CTCGreedyDecoder(config['table_path'])
elif args.post == 'beam_search':
    postprocess = CTCBeamSearchDecoder(config['table_path'])
else:
    postprocess = None

#################################################################################################
model_old = tf.keras.models.load_model(args.structure, compile=False, custom_objects={'BilinearInterpolation': BilinearInterpolation})
if args.weight: model_old.load_weights(args.weight)

input_tensor=model_old.input
ctc_logits=model_old.get_layer('ctc_logits').output
print('Post Processing', args.post)
if args.post == None: output=ctc_logits
else: output=postprocess(ctc_logits)
model = tf.keras.Model(inputs=input_tensor, outputs=output)
for i in range(4): print('successfully load keras model')

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 48, None, 3], model.inputs[0].dtype))
# tf.saved_model.save(model, os.path.join(args.output_dir, 'pb_dir'), signatures=concrete_func)
model.save(os.path.join(args.output_dir, 'pb_dir'), save_format="tf", signatures=concrete_func)

for i in range(4): print('successfully save pb file')
del model

#################################################################################################
# Testing Load and Inference pb graph
# loaded=tf.saved_model.load(os.path.join(args.output_dir, 'pb_dir'))
# print(list(loaded.signatures.keys()))
# infer = loaded.signatures[list(loaded.signatures.keys())[0]]
# print(infer.structured_outputs)

# # pb Inference
# image = cv2.imread('../data/Merge/GRMN1753_1_1.jpg')[...,::-1]
# print(image.shape)
# w,h=image.shape[:2]
# h=int(48.0/w*h)
# w=48
# image = cv2.resize(image, (h,w), interpolation=cv2.INTER_AREA)
# print(image.shape)
# image = (np.expand_dims(image, axis=0) / 255.0).astype(np.float32) # (1,14,63,3)
# result = loaded(image)
# print("successfully do inference in savemodel")
# print(np.argmax(result, axis=-1))
# print("CRNN has {} trainable variables, ...".format(len(loaded.trainable_variables)))
# for v in loaded.trainable_variables:
#     print(v.name)
#################################################################################################


# Convert the model.
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(args.output_dir, 'pb_dir'))
tflite_model = converter.convert()
for i in range(10): print('successfully convert tflite model')

# Save the model.
with open(os.path.join(args.output_dir, 'tflite_dir','model.tflite'), 'wb') as f:
  f.write(tflite_model)
f.close()
for i in range(10): print('successfully save tflite model')
#################################################################################################