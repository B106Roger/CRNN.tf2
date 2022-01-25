import argparse
import pprint
import shutil
import os
from pathlib import Path
import tensorflow as tf
import yaml
from dataset_factory import DatasetBuilder
from losses import LossBox
from metrics import BoxAccuracy
from models import build_pure_stn, NoOpQuantizeConfig, apply_quantization_to_all
from callbacks.callbacks import ImageCallback
from layers.stn import BilinearInterpolation
import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, help='The config file path.')
parser.add_argument('--save_dir', type=Path, required=True, help='The path to save the models, logs, etc.')
parser.add_argument('--weight', type=str, default='', required=False, help='The pretrained weight of model.')
parser.add_argument('--qat', dest='qat', default=False, action="store_true", help="determine whether to use quantilization aware training or not.")
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
tf.config.optimizer.set_jit(True)

#################################################
### Save All Tranining Script                 ###
### for the purpose of reproducing experiment ###
#################################################
with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
pprint.pprint(config)
for filename in ['models.py', 'dataset_factory.py', 'losses.py', 'train_stn.py']:
    shutil.copyfile(f'./crnn/{filename}', f'{args.save_dir}/configs/{filename}')
shutil.copyfile(args.config, f'{args.save_dir}/configs/config.yml')
args.save_dir.mkdir(exist_ok=True)
shutil.copy(args.config, args.save_dir / args.config.name)


################################################
#### Set Up Training Hyper-Parameter       #####
################################################
batch_size = config['batch_size_per_replica']
print(config['dataset_builder'])
dataset_builder = DatasetBuilder(**config['dataset_builder'], require_coords=True, location_only=True)
train_ds = dataset_builder(config['train_ann_paths'], batch_size, True, slice=False)
# train_ds = train_ds.repeat(2)
val_ds = dataset_builder(config['val_ann_paths'], batch_size, False, slice=False)

img_shape=config['dataset_builder']['img_shape']
interpolate_shape=(img_shape[0],img_shape[0])
lr=config['lr_schedule']['initial_learning_rate']
opt=tf.keras.optimizers.Adam(lr)

stn_model, vis_model=build_pure_stn(img_shape=img_shape, interpolation_size=interpolate_shape, model_type=2)
if args.qat:
    stn_model.summary()
    with tfmot.quantization.keras.quantize_scope({
        'NoOpQuantizeConfig': NoOpQuantizeConfig,
        'BilinearInterpolation':BilinearInterpolation,
        }):
        stn_model=tf.keras.models.clone_model(
            stn_model,
            clone_function=apply_quantization_to_all
        )
        stn_model=tfmot.quantization.keras.quantize_apply(stn_model)
stn_model.compile(optimizer=opt, loss=[LossBox()], metrics=[BoxAccuracy()])
stn_model.save(os.path.join(args.save_dir, 'weights', 'structure.h5'), include_optimizer=False)
stn_model.summary()

##############################################
####    Setup Model Training Callback   ######
##############################################
best_model_path = f'{args.save_dir}/weights/best_model.h5'
best_acc_path = f'{args.save_dir}/weights/best_acc.h5'
model_prefix = '{epoch}_{val_loss:.4f}_{val_box_accuracy:.4f}'
model_path = f'{args.save_dir}/weights/{model_prefix}.h5'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_loss', save_weights_only=True, save_best_only=True),
    tf.keras.callbacks.ModelCheckpoint(best_acc_path, monitor='val_box_accuracy', save_weights_only=True, save_best_only=True, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, period=10),
    tf.keras.callbacks.TensorBoard(log_dir=f'{args.save_dir}/logs', **config['tensorboard']),
    ImageCallback(f'{args.save_dir}/images/', train_ds, vis_model, require_coords=True, slice=False, location_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.318, patience=15, min_lr=1e-8, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=51),
]

stn_model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks, validation_data=val_ds,\
    use_multiprocessing=False, workers=4)
