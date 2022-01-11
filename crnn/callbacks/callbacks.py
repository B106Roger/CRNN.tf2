import tensorflow as tf
import os
import cv2 
import numpy as np
import shutil
from tensorflow import keras

class ImageCallback(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, folder, dataset, stn_model, require_coords, slice, row=8, count=2, location_only=False):
        super(ImageCallback, self).__init__()
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        self.folder = folder
        self.dataset = dataset
        self.stn_model = stn_model
        self.require_coords = require_coords
        self.slice = slice
        self.row=row
        self.count = count
        self.location_only=location_only
    
    def get_predict_point(self, transform_mat):
        """
        transform_mat: (batch, 6)
        """
        transform_mat = transform_mat.reshape((-1, 2, 3))
        my_coord = np.array([[
            [-1,-1,1],
            [ 1, 1,1],
            [ 1,-1,1],
            [-1, 1,1]
        ]])
        my_coord = my_coord.transpose((0,2,1))
        new_coord = np.matmul(transform_mat, my_coord)
        new_coord = new_coord.transpose((0,2,1))
        return new_coord

    def coord_to_int(self, coords, imgshape):
        b, ih, iw = imgshape[:3]
        n_points=coords.shape[-1] // 2
        ncoords = (coords + 1.0) / 2.0 * np.array([[iw, ih]*n_points])
        ncoords = ncoords.astype(np.int32).reshape((b, -1))
        return ncoords

    def on_epoch_end(self, epoch, logs={}):
        for i, (images,labels) in enumerate(self.dataset, 1):
            if self.require_coords:
                if self.slice:
                    if self.location_only:
                        coords, slice=labels
                    else:
                        label, coords, slice=labels
                else:
                    if self.location_only:
                        coords = labels[0]
                    else:
                        label, coords=labels
            stn_result, transform_mat = self.stn_model(images, training=False)
            # process origin image
            images = images.numpy()[:self.row]
            images = (images*255.).astype(np.uint8)
            transform_mat = transform_mat.numpy()[:self.row]

            pcoords = self.coord_to_int(self.get_predict_point(transform_mat), images.shape)
            if self.require_coords: 
                gcoords = self.coord_to_int(coords.numpy()[:self.row], images.shape)
            n_points = pcoords.shape[-1] // 2
            for ii in range(len(images)):
                img = images[ii].copy()
                for iii in range(n_points):
                    if self.require_coords: 
                        images[ii] = cv2.circle(img, tuple(gcoords[ii,2*iii:2*(iii+1)]), 3, (int(127+128/4*iii), 0, 0), -1)
                    images[ii] = cv2.circle(img, tuple(pcoords[ii,2*iii:2*(iii+1)]), 3, (0, 0, int(127+128/4*iii)), -1)
                    
            images = np.vstack(images)
            # process stn_result
            stn_result = stn_result.numpy()[:self.row]
            stn_result = (stn_result*255.).astype(np.uint8)
            stn_result = np.vstack(stn_result)
            h, w = stn_result.shape[:2]
            w = int(w * images.shape[0] / h)
            h = int(images.shape[0])
            stn_result = cv2.resize(stn_result, (w, h))
            stn_result[:,0]=255
            # total image
            filename = f'epoch_{epoch}_{i}.png'
            show_result = np.concatenate([images, stn_result], axis=1)
            cv2.imwrite(os.path.join(self.folder, filename), show_result[...,::-1])
            if i == self.count: break

class ImageCallback2(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, folder, dataset, model_vis, require_coords, slice=False, row=8, count=2, location_only=False):
        super(ImageCallback2, self).__init__()
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        self.folder = folder
        self.dataset = dataset
        self.model_vis = model_vis
        self.require_coords = require_coords
        self.slice = slice
        self.row=row
        self.count = count
        self.location_only=location_only
    

    def on_epoch_end(self, epoch, logs={}):
        if epoch %10 != 0: return 
        for i, (images,labels) in enumerate(self.dataset, 1):
            if self.require_coords:
                if self.slice:
                    if self.location_only:
                        coords, slice=labels
                    else:
                        label, coords, slice=labels
                else:
                    if self.location_only:
                        coords = labels[0]
                    else:
                        label, coords=labels
            else:
                label=labels[0]
            print('images',images.shape)
            label=tf.sparse.to_dense(label).numpy()
            pred_res, stn_result = self.model_vis(images, training=False)
            # for _i in range(len(pred_res)):
            #     for _t in range(4):
            #         for clss in range(12):
            #             print(f'{pred_res[_i,_t,clss]:6.3f} ', end='')
                    
            #         print(f'{np.argmax(pred_res[_i,_t])} ')
            #     print()
            # # tf.print('pred_res', pred_res)
            # tf.print('gth', label)
            # process origin image
            images = images.numpy()[:self.row]
            images = (images*255.).astype(np.uint8)
            height, width=images.shape[1:3]

            pred_res=tf.math.argmax(pred_res, axis=-1).numpy()
            label_canvas=np.repeat(np.ones_like(images), 2, axis=2)
            
            for ii in range(len(images)):
                
                pre='-'.join([str(ele) for ele in pred_res[ii]])
                lab='-'.join([str(ele) for ele in label[ii]])
                cv2.putText(label_canvas[ii], f'g: {lab}', (0,12),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(label_canvas[ii], f'p: {pre}', (0,height-24+12),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            
            label_canvas = np.vstack(label_canvas)
            images = np.vstack(images)
            # process stn_result
            stn_result = stn_result.numpy()[:self.row]
            stn_result = (stn_result*255.).astype(np.uint8)
            stn_result = np.vstack(stn_result)
            h, w = stn_result.shape[:2]
            w = int(w * images.shape[0] / h)
            h = int(images.shape[0])
            stn_result = cv2.resize(stn_result, (w, h))
            stn_result[:,0]=255
            label_canvas[:,0]=255
            # total image
            filename = f'epoch_{epoch}_{i}.png'
            
            show_result = np.concatenate([images, stn_result, label_canvas], axis=1)
            cv2.imwrite(os.path.join(self.folder, filename), show_result[...,::-1])
            if i == self.count: break


class ModelWeight(keras.callbacks.Callback):
    def __init__(self, layername):
        super().__init__()
        self.layername=layername
    def on_epoch_end(self, epoch, logs=None):
        weights=self.model.get_layer(self.layername).get_weights()
        print(tf.reshape(weights[1], (-1)))

