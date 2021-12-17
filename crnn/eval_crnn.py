import argparse
import pprint
import tensorflow as tf
import yaml
import glob
import os

from dataset_factory import DatasetBuilder, DatasetBuilderV2
from losses import CTCLoss, LossBox
from metrics import BoxAccuracy, SequenceAccuracy, EditDistance
from layers.stn import BilinearInterpolation



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False, help='The config file path.')
parser.add_argument('--weight', type=str, required=True, help='The saved weight path.(.h5 weight file)')
parser.add_argument('--structure', type=str, required=False, help='The saved structure path.(.h5 weight file which include structure)')
parser.add_argument('--point4', dest='point4', default=False, action='store_true', help="Require box accuracy as one of the metric")

args = parser.parse_args()
if args.structure is None:
    args.structure = os.path.join(os.path.dirname(args.weight), 'structure.h5')
if args.config is None:
    print(os.path.join(os.path.dirname(args.weight), '*.yml'))
    default_list = glob.glob(os.path.join(os.path.dirname(args.weight), '..', '*.yml'))
    if len(default_list) == 0: 
        print('[Warning] No Config File Found in Directory of Model Weight.')
        exit()
    elif len(default_list) == 1:
        args.config=default_list[0]
    else:
        print('[Warning] Multiple Config File Found in the Directory of Model Weight.')
        args.config=default_list[0]
    print('Loading Config File: ', args.config)

with open(args.config) as f:
    parse_config = yaml.load(f, Loader=yaml.Loader)
    config = parse_config['eval']
    val_conf = parse_config['train']
pprint.pprint(config)

dataset_builder = DatasetBuilderV2(**config['dataset_builder'], require_coords=args.point4)
ds = dataset_builder(config['ann_paths'], 1, False)
train_ds = dataset_builder(val_conf['train_ann_paths'], val_conf['batch_size_per_replica'], False)
val_ds = dataset_builder(val_conf['val_ann_paths'], val_conf['batch_size_per_replica'], False)
model = tf.keras.models.load_model(args.structure, custom_objects={
    'BilinearInterpolation': BilinearInterpolation
}, compile=False)
model.load_weights(args.weight)



########################################################
########     Reconstruct Model             #############
########################################################
inputs=model.layers[0].input
if args.point4:
    outputs1=model.get_layer('ctc_logits').output
    outputs2=model.get_layer('stn').output
    model = tf.keras.Model(inputs, [outputs1, outputs2])
else:
    outputs1=model.get_layer('ctc_logits').output
    model = tf.keras.Model(inputs, outputs1)
model.summary()

########################################################
########    Setup Metric for Model           ###########
########################################################
if args.point4:
    loss_dict={ 
        model.output_names[0]: [CTCLoss()],
        model.output_names[1]: [LossBox()] 
    }
    metrics_dict={ 
        model.output_names[0]: [SequenceAccuracy(), EditDistance()],
        model.output_names[1]: [BoxAccuracy(iou_threshold=0.8)]
    }
else:
    loss_dict=[CTCLoss()]
    metrics_dict=[SequenceAccuracy(), EditDistance()]


model.compile(loss=loss_dict, metrics=metrics_dict)
print('Verify Model Accuracy in Training data')
model.evaluate(train_ds)
print('Verify Model Accuracy in Validation data')
model.evaluate(val_ds)
