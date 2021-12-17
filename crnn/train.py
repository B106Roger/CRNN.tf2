import argparse
import json
import pprint
import shutil
import os
from pathlib import Path
import tensorflow as tf
import yaml
from dataset_factory import DatasetBuilder, DatasetBuilderV2
from losses import CTCLoss, LossBox, SliceLoss
from metrics import SequenceAccuracy
from models import build_model
from callbacks.callbacks import ImageCallback, ImageCallback2

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='The config file path.')
parser.add_argument('--save_dir', type=str, required=True, help='The path to save the models, logs, etc.')
parser.add_argument('--weight', type=str, default='', required=False, help='The pretrained weight of model.')
parser.add_argument('--ext_bg_ratio', type=float, default=0.7, help="use to specify that the ratio of background which training image should include ")

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--use_stn', dest='use_stn', action='store_true')
feature_parser.add_argument('--no-stn', dest='use_stn', action='store_false')
parser.set_defaults(use_stn=False)
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--train_stn', dest='train_stn', action='store_true')
feature_parser.add_argument('--no-train_stn', dest='train_stn', action='store_false')
parser.set_defaults(train_stn=False)
parser.add_argument("--slice", default=False, action="store_true")
args = parser.parse_args()


os.makedirs(f'{args.save_dir}/weights', exist_ok=True)
os.makedirs(f'{args.save_dir}/configs', exist_ok=True)

#############################
#### Specify GPU usuage #####
#############################
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    print(gpus)
    try:
        for i in range(len(gpus)):
            mem = 1024 * 8 if i == 0 else 1024 * 10
            tf.config.set_visible_devices(gpus[i], 'GPU')
            tf.config.set_logical_device_configuration(gpus[i], [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
tf.config.optimizer.set_jit(True)

#################################################
### Save All Tranining Script                 ###
### for the purpose of reproducing experiment ###
#################################################
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)

for filename in ['models.py', 'dataset_factory.py', 'losses.py', 'train.py']:
    shutil.copyfile(f'./crnn/{filename}', f'{args.save_dir}/configs/{filename}')
shutil.copyfile(args.config, f'{args.save_dir}/configs/config.yml')
os.makedirs(args.save_dir, exist_ok=True)
shutil.copy(args.config, os.path.join(args.save_dir, os.path.basename(args.config)))
with open(os.path.join(f'{args.save_dir}/configs/training_argument.json'), 'w') as f:
    dict_args=vars(args)
    print(dict_args)
    json.dump(dict_args, f)


################################################
#### Set Up Training Hyper-Parameter       #####
################################################
batch_size = config['batch_size_per_replica']
print(config['dataset_builder'])
dataset_builder = DatasetBuilderV2(**config['dataset_builder'], require_coords=args.train_stn)
train_ds = dataset_builder(config['train_ann_paths'], batch_size, is_training=True,  slice=args.slice, background_ratio=0.15,\
     ignore_unknown=True, ext_bg_ratio=args.ext_bg_ratio)
val_ds =   dataset_builder(config['val_ann_paths'],           16, is_training=False, slice=args.slice, background_ratio=0.0)

model, callback_model = build_model(dataset_builder.num_classes,
                    use_stn=args.use_stn,
                    train_stn=args.train_stn,
                    slice=args.slice,
                    weight=args.weight,
                    img_shape=config['dataset_builder']['img_shape'])
lr=config['lr_schedule']['initial_learning_rate']
# opt=tf.keras.optimizers.Adam(lr)
opt=tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
losses_list=[[CTCLoss()],[LossBox()],[SliceLoss()]]
if args.slice: losses_list.pop()
metrics_list={'ctc_logits': SequenceAccuracy()}
model.compile(optimizer=opt, loss=losses_list, metrics=metrics_list)
model.save(os.path.join(args.save_dir, 'weights', 'structure.h5'), include_optimizer=False)
model.summary()



##############################################
####    Setup Model Training Callback   ######
##############################################
best_model_path = f'{args.save_dir}/weights/best_model.h5'
best_acc_path = f'{args.save_dir}/weights/best_acc.h5'
model_prefix = '{epoch}_{val_loss:.4f}_{val_ctc_logits_sequence_accuracy:.4f}' if (args.use_stn and args.train_stn)\
     else '{epoch}_{val_loss:.4f}_{val_sequence_accuracy:.4f}'
ckpt_prefix='val_ctc_logits_sequence_accuracy' if (args.use_stn and args.train_stn)\
     else 'val_sequence_accuracy'
model_path = f'{args.save_dir}/weights/{model_prefix}.h5'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_loss', save_weights_only=True, save_best_only=True),
    tf.keras.callbacks.ModelCheckpoint(best_acc_path, monitor=ckpt_prefix, save_weights_only=True, save_best_only=True, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, period=10),
    tf.keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs', **config['tensorboard']),
    ImageCallback2(f'{args.save_dir}/images/', train_ds, callback_model, require_coords=args.train_stn, slice=args.slice),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.318, patience=15, min_lr=1e-8, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=51),
]
model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks, validation_data=val_ds,\
    use_multiprocessing=False, workers=4)
