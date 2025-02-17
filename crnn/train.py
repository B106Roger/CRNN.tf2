import argparse
import json
import pprint
import shutil
import os
from pathlib import Path
import tensorflow as tf
import yaml
from dataset_factory import DatasetBuilderV2
from losses import CTCLoss, LossBox
from metrics import SequenceAccuracy
from layers.stn import BilinearInterpolation
from models import build_model, apply_quantization_to_all, NoOpQuantizeConfig
from callbacks.callbacks import ImageCallback2
import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='The config file path.')
parser.add_argument('--save_dir', type=str, required=True, help='The path to save the models, logs, etc.')
parser.add_argument('--weight', type=str, default='', required=False, help='The pretrained weight of model.')
parser.add_argument('--ext_bg_ratio', type=float, default=0.4, help="use to specify that the ratio of background which training image should include ")
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--use_stn', dest='use_stn', action='store_true')
feature_parser.add_argument('--no_stn', dest='use_stn', action='store_false')
parser.set_defaults(use_stn=False)
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--train_stn', dest='train_stn', action='store_true')
feature_parser.add_argument('--no_train_stn', dest='train_stn', action='store_false')
parser.set_defaults(train_stn=False)
parser.add_argument('--qat', default=False, action="store_true", help="determine whether to use quantilization aware training or not.")
args = parser.parse_args()


if os.path.exists(args.save_dir):
    print("""
******************************************
****************  Warning ****************
******************************************
The folder already exist, do you really 
want overwrite the content of the folder ?
""")
    choice=input("Y/N: ")
    if choice[0] == "N" or choice[0] == "n":
        print("choose another folder name please")
        exit()
os.makedirs(f'{args.save_dir}/weights', exist_ok=True)
os.makedirs(f'{args.save_dir}/configs', exist_ok=True)

#############################
#### Specify GPU usuage #####
#############################
gpus = tf.config.list_physical_devices('GPU')
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

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
train_ds=dataset_builder(config['train_ann_paths'], batch_size, is_training=True,  background_ratio=0.2, ext_bg_ratio=args.ext_bg_ratio, ignore_unknown=True)
val_ds  =dataset_builder(config['val_ann_paths'],          128, is_training=False, background_ratio=0.2, ext_bg_ratio=args.ext_bg_ratio)

model, callback_model = build_model(dataset_builder.num_classes,
                    use_stn=args.use_stn,
                    train_stn=args.train_stn,
                    weight=args.weight,
                    img_shape=config['dataset_builder']['img_shape'])
lr=config['lr_schedule']['initial_learning_rate']
# opt=tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
opt=tf.keras.optimizers.Adam(lr)
losses_list=[[CTCLoss()],[LossBox()]]
# metrics_list={'ctc_logits': SequenceAccuracy()}
metrics_list=[[SequenceAccuracy()],[]]
if args.qat:
    with tfmot.quantization.keras.quantize_scope({
        'NoOpQuantizeConfig': NoOpQuantizeConfig,
        'BilinearInterpolation':BilinearInterpolation,
    }):
        model=tf.keras.models.clone_model(
            model,
            clone_function=apply_quantization_to_all
        )
        model=tfmot.quantization.keras.quantize_apply(model)
model.compile(optimizer=opt, loss=losses_list, metrics=metrics_list)
model.save(os.path.join(args.save_dir, 'weights', 'structure.h5'), include_optimizer=False)
model.summary()



##############################################
####    Setup Model Training Callback   ######
##############################################
quant_prefix='quant_' if args.qat else ''
best_model_path = f'{args.save_dir}/weights/best_model.h5'
best_acc_path = f'{args.save_dir}/weights/best_acc.h5'
model_prefix = '{epoch}_{val_loss:.4f}_{val_%sctc_logits_sequence_accuracy:.4f}'%(quant_prefix) if (args.use_stn and args.train_stn)\
     else '{epoch}_{val_loss:.4f}_{val_%ssequence_accuracy:.4f}'%(quant_prefix)
ckpt_prefix='val_%sctc_logits_sequence_accuracy'%(quant_prefix) if (args.use_stn and args.train_stn)\
     else 'val_%ssequence_accuracy'%(quant_prefix)
model_path = f'{args.save_dir}/weights/{model_prefix}.h5'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_loss', save_weights_only=True, save_best_only=True),
    tf.keras.callbacks.ModelCheckpoint(best_acc_path, monitor=ckpt_prefix, save_weights_only=True, save_best_only=True, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, period=10),
    tf.keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs', **config['tensorboard']),
    ImageCallback2(f'{args.save_dir}/images/', train_ds, callback_model, require_coords=args.train_stn),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.318, patience=10, min_lr=1e-8, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=51),
]
# train_ds=train_ds.repeat(2)
model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks, validation_data=val_ds,\
    use_multiprocessing=False, workers=4)
