import argparse
from pathlib import Path
import cv2
import yaml
import tensorflow as tf
import os
import numpy as np

print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--pre', type=str, help='pre processing.')
parser.add_argument('--weight',type=str, required=True, help='directory of weight')
parser.add_argument('--export_dir',type=str, required=True, help='directory of export dir')
parser.add_argument('--fp32', default=False, action="store_true")
parser.add_argument('--fp16', default=False, action="store_true")
parser.add_argument("--int8", default=False, action="store_true")

args = parser.parse_args()
os.makedirs(args.export_dir, exist_ok=True)

# Specify GPU usuage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    print(gpus)
    try:
        for i in range(len(gpus)):
            mem = 1024 * 4
            tf.config.set_visible_devices(gpus[i], 'GPU')
            tf.config.set_logical_device_configuration(gpus[i], [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Print Parameters 
tf.keras.models.load_model(args.weight).summary()

# Testing Load and Inference pb graph
loaded=tf.saved_model.load(args.weight)
print(list(loaded.signatures.keys()))
infer = loaded.signatures[list(loaded.signatures.keys())[0]]
print('infer.structured_input_signature', infer.structured_input_signature)
print('infer.structured_outputs', infer.structured_outputs)
INPUT_SHAPE=tuple(infer.structured_input_signature[1]['x'].shape)
# pb Inference
image = np.zeros(INPUT_SHAPE).astype(np.float32)
print('input image detaial', image.shape, image.dtype)
result = loaded(image)
print("successfully do inference in savemodel")
print('result of output', result.shape, result.dtype)
print("CRNN has {} trainable variables, ...".format(len(loaded.trainable_variables)))
# for v in loaded.trainable_variables:
#     print(v.name)

if args.fp32:
    ##########################################
    # Load save_model and Convert to TFlite  #
    ##########################################
    converter = tf.lite.TFLiteConverter.from_saved_model(args.weight)
    tflite_model = converter.convert()
    print('successfully convert tflite fp32 model')

    # Save the model.
    with open(os.path.join(args.export_dir, 'model_fp32.tflite'), 'wb') as f:
      f.write(tflite_model)
    f.close()
    print('successfully save tflite fp32 model')
    
    ##########################################
    # Load TFlite model and inference        #
    ##########################################
    interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, 'model_fp32.tflite'))
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print('fp32 input_details', input_details)
    print('fp32 output_details', output_details)

    # Inference Method 1
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details["index"], np.zeros(INPUT_SHAPE, dtype=np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    print('fp32 output result', output.shape, output.dtype)

    # Incerence Method 2
    # my_signature = interpreter.get_signature_runner()
    # # my_signature is callable with input as arguments.
    # output = my_signature(x=tf.constant([1.0], shape=INPUT_SHAPE, dtype=tf.float32))
    # print(output.keys())
    # # 'output' is dictionary with all outputs from the inference.
    # # In this case we have single output 'result'.
    # print(output['output_0'].shape, output['output_0'].dtype)

    print('successfully inference in tflite fp32 model')

if args.fp16:
    ##########################################
    # Load save_model and Convert to TFlite  #
    ##########################################
    converter = tf.lite.TFLiteConverter.from_saved_model(args.weight)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    print('successfully convert tflite fp16 model')

    # Save the model.
    with open(os.path.join(args.export_dir, 'model_fp16.tflite'), 'wb') as f:
      f.write(tflite_model)
    f.close()
    print('successfully save tflite fp16 model')
    
    ##########################################
    # Load TFlite model and inference        #
    ##########################################
    interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, 'model_fp16.tflite'))
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print('fp16 input_details', input_details)
    print('fp16 output_details', output_details)

    # Inference Method 1
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details["index"], np.zeros(INPUT_SHAPE, dtype=np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    print('fp16 output result', output.shape, output.dtype)

    # Incerence Method 2
    # my_signature = interpreter.get_signature_runner()
    # # my_signature is callable with input as arguments.
    # output = my_signature(x=tf.constant([1.0], shape=INPUT_SHAPE, dtype=tf.float32))
    # print(output.keys())
    # # 'output' is dictionary with all outputs from the inference.
    # # In this case we have single output 'result'.
    # print(output['output_0'].shape)

    print('successfully inference in tflite fp16 model')

if args.int8:
    ##########################################
    # Load save_model and Convert to TFlite  #
    ##########################################
    train_images=np.zeros((200, INPUT_SHAPE[1],INPUT_SHAPE[2],INPUT_SHAPE[3]), dtype=np.float32)
    def representative_data_gen():
      for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]
      
    converter = tf.lite.TFLiteConverter.from_saved_model(args.weight)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_enable_resource_variables = True
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,
      tf.lite.OpsSet.SELECT_TF_OPS,
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converted_model_name='model_int8_inpuint8_outuint8.tflite'
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    print('successfully convert tflite int8 model')

    # Save the model
    with open(os.path.join(args.export_dir, converted_model_name), 'wb') as f:
      f.write(tflite_model)
    f.close()
    print('successfully save tflite int8 model')


    ##########################################
    # Load TFlite model and inference        #
    ##########################################
    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(os.path.join(args.export_dir, converted_model_name))
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print('int8 input_details', input_details)
    print('int8 output_details', output_details)
    
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details["index"], np.zeros(INPUT_SHAPE, dtype=input_details['dtype']))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    print('int8 output result', output.shape, output.dtype)
    print('successfully inference in tflite int8 model')


