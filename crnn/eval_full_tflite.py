import argparse
from tqdm import tqdm
import shutil

import tensorflow as tf
import yaml
import os
import numpy as np
import cv2
import tensorflow_addons as tfa
import time
from models import InferenceModel
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder
from dataset_factory import DatasetBuilder
from metrics import BoxAccuracy, SequenceAccuracy, SequenceTop5Accuracy
from layers.stn import BilinearInterpolation

def save_result(img, result, decoder, label, prefix='1', ):
    """
    img: (b, 64,320)
    result: (b, 4, 12)
    """
    img=(img*255.0).astype(np.uint8)
    result, scores=decoder(result)
    scores = scores.numpy()
    label=tf.sparse.to_dense(label, -1)
    
    for i, (a_img, a_result, score, a_label) in enumerate(zip(img, result, scores, label)):
        h=a_img.shape[0]
        a_canvas=np.zeros((h, h//2, 3), np.uint8)

        word=a_result.numpy()
        word=str(word.decode('utf-8')) 
        a_label=a_label.numpy()
        a_label=[str(item-1)  for item in a_label if item != -1]
        a_label=''.join(a_label)
        # print('scores', scores)
        # Put Prediction Result
        cv2.putText(a_canvas, word, (0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255))
        cv2.putText(a_canvas, f'{score:4.2f}', (0,40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255))
        
        # Put Ground Truth
        cv2.rectangle(a_img, (0,0), (30,10), (0,0,0), -1)
        cv2.putText(a_img, a_label, (0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255))
        
        a_img=np.concatenate([a_img, a_canvas], axis=1)
        # Correct
        correct = 'true' if word == a_label else 'false'
        cv2.imwrite(f'demo/{correct}/{prefix}_{i}.jpg', a_img[...,::-1])

def evaluation(crnn_model, model_aspect_ratio, dataset, bin=[115,256,576], visualize=False, decoder=None, point4=True):
    seq_metric=SequenceTop5Accuracy()
    # seq_metric=SequenceAccuracy()
    box_metric=BoxAccuracy()
    bin_of_correct=np.zeros((len(bin)+1,), dtype=np.int32)
    bin_of_wrong=np.zeros((len(bin)+1,), dtype=np.int32)

    #####################################
    #          | crnn_true | crnn_false #
    # _________|___________|___________ #
    # stn_true |           |            #
    # _________|___________|___________ #
    # stn_false|           |            #
    # _________|___________|___________ #
    #####################################
    confusion_matrix=np.zeros((2,2), np.int32)

    bin=np.array(bin, np.float32)
    bin=np.insert(bin, 0, 0)
    bin=np.append(bin, np.inf)
    
    batch_num=1
    cnt=0
    batch_shapes=[]
    batch_images=np.zeros((batch_num, 64, int(64*model_aspect_ratio), 3), dtype=np.float32)
    batch_labels=[]
    batch_coords=[]

    for i, (images,(label, coord)) in tqdm(enumerate(dataset), ncols=150):
        inference_images=tf.image.resize(images, (64, int(64*model_aspect_ratio)))

        batch_images[cnt]=inference_images[0]
        batch_shapes.append(images[0].shape)
        batch_labels.append(label)
        batch_coords.append(coord)
        cnt+=1
        
        if cnt == batch_num: 
            batch_labels=tf.sparse.concat(0, batch_labels, expand_nonconcat_dims =True)
            batch_coords=tf.concat(batch_coords, axis=0)
        else: continue

        start_time=time.time()
        crnn_res=crnn_model(batch_images)
        # print('elipse time: ', time.time() - start_time)
        if visualize:
            save_result(batch_images, crnn_res, decoder, batch_labels, i)

        seq_correct=seq_metric.update_state(batch_labels, crnn_res).numpy().astype(np.int32)
        for ii in range(len(seq_correct)):
            confusion_matrix[0, 1-seq_correct[ii]] += 1
            h, w=batch_shapes[ii][:2]
            normalize_w=(batch_coords[ii,2]-batch_coords[ii,0])/2.0
            normalize_h=(batch_coords[ii,3]-batch_coords[ii,1])/2.0
            area=(h*normalize_h)*(w*normalize_w)
            for jj in range(len(bin)-1):
                if bin[jj] <= area and area < bin[jj+1]:
                    if seq_correct[ii] == 1:
                        bin_of_correct[jj]+=1
                    else:
                        bin_of_wrong[jj]+=1
    
        cnt = 0
        batch_labels=[]
        batch_coords=[]
        batch_shapes=[]


    if cnt > 0:
        # concat label
        batch_images=batch_images[:cnt]
        batch_labels=tf.sparse.concat(0, batch_labels, expand_nonconcat_dims =True)
        batch_coords=tf.concat(batch_coords, axis=0)
        # model
        crnn_res=crnn_model(batch_images)
        # metric
        seq_correct=seq_metric.update_state(batch_labels, crnn_res).numpy().astype(np.int32)
        if visualize:
            save_result(batch_images, crnn_res, decoder, batch_labels, i)
        for ii in range(len(seq_correct)):
            confusion_matrix[0, 1-seq_correct[ii]] += 1
            h, w=batch_shapes[ii][:2]
            normalize_w=batch_coords[ii,2]-batch_coords[ii,0]
            normalize_h=batch_coords[ii,3]-batch_coords[ii,1]
            area=(h*normalize_h)*(w*normalize_w)
            for jj in range(len(bin)-1):
                if bin[jj] < area and area < bin[jj+1]:
                    if seq_correct[ii] == 1:
                        bin_of_correct[jj]+=1
                    else:
                        bin_of_wrong[jj]+=1
                        
    # Print Title Bar
    for i in range(len(bin)):
        if i == 0:
            print('%12s'%'Area | ', end='')
        else:
            print(f'{bin[i-1]:5.0f} ~ {bin[i]:5.0f} | ', end='')
    print()
    # Print Correct Content
    for i in range(len(bin)):
        if i == 0:
            print('%12s'%'Correct | ', end='')
        else:
            print(f'{bin_of_correct[i-1]:13d} | ', end='')
    print()
    # Print Wrong Content
    for i in range(len(bin)):
        if i == 0:
            print('%12s'%'Wrong | ', end='')
        else:
            print(f'{bin_of_wrong[i-1]:13d} | ', end='')
    print()

    # Print Accuracy
    for i in range(len(bin)):
        if i == 0:
            print('%12s'%'Accuracy | ', end='')
        else:
            total=bin_of_correct[i-1]+bin_of_wrong[i-1]
            acc=0.0
            if total!=0:
                acc=bin_of_correct[i-1]/total
            acc*=100.0
            print(f'{acc:12.2f}% | ', end='')
    print()

    print(f'''
#####################################
# iou>{0.5:4.2f} | crnn_true | crnn_false #
# _________|___________|___________ #
# stn_true |{confusion_matrix[0,0]:8d}   |{confusion_matrix[0,1]:8d}    #
# _________|___________|___________ #
# stn_false|{confusion_matrix[1,0]:8d}   |{confusion_matrix[1,1]:8d}    #
# _________|___________|___________ #
#####################################
    ''')

    print(f'seq_acc: {seq_metric.result():8.6f}')
    return bin_of_correct, bin_of_wrong, confusion_matrix


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='The config file path.')
    parser.add_argument('--crnn_weight', type=str, default='', required=False, help='The merged crnn path.(tflite and savemodel format supported')

    args = parser.parse_args()

    with open(args.config) as f:
        parse_config = yaml.load(f, Loader=yaml.Loader)
        config = parse_config['eval']
        val_conf = parse_config['train']

    # Set the image shape to None, so that the loader would load origin image without resize it.
    config['dataset_builder']['img_shape']=(None, None, 3)
    print(config)
    

    shutil.rmtree('demo', ignore_errors=True)
    os.makedirs('demo', exist_ok=True)
    os.makedirs('demo/true', exist_ok=True)
    os.makedirs('demo/false', exist_ok=True)
    # exit()

    #######################################
    ######## Build Dataset        #########
    #######################################
    dataset_builder = DatasetBuilder(**config['dataset_builder'], require_coords=True)
    train_ds = dataset_builder(val_conf['train_ann_paths'], 1, False, ignore_unknown=True)
    val_ds = dataset_builder(val_conf['val_ann_paths'], 1, False, ignore_unknown=True)

    #######################################
    ######## Load Model            ########
    #######################################
    crnn_model=InferenceModel(args.crnn_weight, 'crnn_model', 'crnn_model')
    # decoder=CTCBeamSearchDecoder(parse_config['dataset_builder']['table_path'])
    decoder=CTCGreedyDecoder(parse_config['dataset_builder']['table_path'])
    
    # w/h
    model_aspect_ratio=crnn_model.input_shape[2] / crnn_model.input_shape[1]


    ##########################################
    ######## Evaluation on Training Data #####
    ##########################################

    # print('\n\ntraining data')
    # correct, wrong, confusion_matrix = evaluation(stn_model, crnn_model, model_aspect_ratio, train_ds, visualize=True, decoder=decoder, point4=args.point4)
    
    print('\n\nvalidation data')
    correct, wrong, confusion_matrix = evaluation(crnn_model, model_aspect_ratio, val_ds, visualize=True, decoder=decoder)