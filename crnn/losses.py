import tensorflow as tf
from tensorflow import keras

def iou(y_true, y_pred):
    """
    y_true: (b, 4)
    y_pred: (b, 4)
    """
    box1_xy1=y_true[:,0:2]
    box1_xy2=y_true[:,2:4]
    box2_xy1=y_pred[:,0:2]
    box2_xy2=y_pred[:,2:4]

    box1_wh=box1_xy2-box1_xy1
    area1=tf.math.maximum(box1_wh[:,0]*box1_wh[:,1], 0.0)

    box2_wh=box2_xy2-box2_xy1
    area2=tf.math.maximum(box2_wh[:,0]*box2_wh[:,1], 0.0)

    inter_xy1 = tf.math.maximum(box1_xy1, box2_xy1)
    inter_xy2 = tf.math.minimum(box1_xy2, box2_xy2)
    inter_wh = inter_xy2 - inter_xy1 
    inter_area = tf.math.maximum(inter_wh[:,0]*inter_wh[:,1], 0.0)
    
    iou = inter_area / (area1+area2-inter_area)
    # res = 1.0 - iou
    
    return iou

def diou_loss(y_true, y_pred):
    """
    y_true: (b, 4)
    y_pred: (b, 4)
    """
    box1_xy1=y_true[:,0:2]
    box1_xy2=y_true[:,2:4]
    box2_xy1=y_pred[:,0:2]
    box2_xy2=y_pred[:,2:4]

    box1_wh=box1_xy2-box1_xy1
    box1_c =(box1_xy1+box1_xy2)/2.0
    area1=tf.math.maximum(box1_wh[:,0]*box1_wh[:,1], 0.0)

    box2_wh=box2_xy2-box2_xy1
    box2_c =(box2_xy1+box2_xy2)/2.0
    area2=tf.math.maximum(box2_wh[:,0]*box2_wh[:,1], 0.0)

    inter_xy1 = tf.math.maximum(box1_xy1, box2_xy1)
    inter_xy2 = tf.math.minimum(box1_xy2, box2_xy2)
    inter_wh = inter_xy2 - inter_xy1 
    inter_area = tf.math.maximum(inter_wh[:,0]*inter_wh[:,1], 0.0)

    iou = tf.keras.backend.switch(inter_area==0, tf.zeros_like(inter_area), inter_area / (area1+area2-inter_area))
    dist = tf.math.sqrt(tf.reduce_sum((box1_c-box2_c)**2, axis=-1))
    res = 1.0 - iou + dist
    
    return res

def average_diou(y_true, y_pred):
    diou1 = diou_loss(y_true[:,:4], y_pred[:,:4])
    # [4,5,6,7]
    y_true_x1y1 = tf.gather(y_true, indices=[6,5,4,7], axis=-1)
    y_pred_x2y2 = tf.gather(y_pred, indices=[6,5,4,7], axis=-1)
    diou2 = diou_loss(y_true_x1y1, y_pred_x2y2)
    return (diou1+diou2) / 2.0

class CTCLoss(keras.losses.Loss):
    """ A class that wraps the function of tf.nn.ctc_loss. 
    
    Attributes:
        logits_time_major: If False (default) , shape is [batch, time, logits], 
            If True, logits is shaped [time, batch, logits]. 
        blank_index: Set the class index to use for the blank label. default is
            -1 (num_classes - 1). 
    """

    def __init__(self, logits_time_major=False, blank_index=-1, 
                 name='ctc_loss', reduction=None):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    @tf.function(experimental_relax_shapes=True)
    def call(self, y_true, y_pred):
        """ Computes CTC (Connectionist Temporal Classification) loss. work on
        CPU, because y_true is a SparseTensor.
        """
        y_pred_shape = tf.shape(y_pred)
        y_true_shape = tf.shape(y_true)
        y_true = tf.sparse.to_dense(y_true, default_value=99)
        y_true = tf.cast(y_true, tf.int32)
        pad=tf.ones_like(y_true)*99
        y_true=tf.concat([y_true,pad], axis=-1)
        label_length = tf.math.argmax(y_true==99, axis=1)
        logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])

        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.math.reduce_mean(loss)

class LossBox(keras.losses.Loss):
    """ A class that wraps the function of tf.nn.ctc_loss. 
    
    Attributes:
        logits_time_major: If False (default) , shape is [batch, time, logits], 
            If True, logits is shaped [time, batch, logits]. 
        blank_index: Set the class index to use for the blank label. default is
            -1 (num_classes - 1). 
    """

    def __init__(self, loss=None, reduction=None, name='LossBox'):
        super().__init__()
        if loss is None:
            self.loss = tf.keras.losses.huber
        else:
            self.loss = loss
        # shape=(4, 3), the 2nd coordinate contain coordinate (x,y,1.0)
        self.stn_points = tf.constant([[ 
            [-1.0, -1.0, 1.0], # Left  Top
            [ 1.0,  1.0, 1.0], # Right Bottom
            [ 1.0, -1.0, 1.0], # Right Top
            [-1.0,  1.0, 1.0], # Left  Bottom
        ]])
        self.stn_points = tf.transpose(self.stn_points, perm=(0,2,1)) # (1, 3, n_points)
    @tf.function(experimental_relax_shapes=True)
    def call(self, y_true, y_pred, *args):
        """
        Calculate the loss for stn coordinate
        Parameters
        ----------
        y_true: (batch, 4)
        y_pred: (batch, 6)
        """
        # Calculate Mask
        mask=tf.math.reduce_all(tf.abs(y_true)<=1e-6, axis=-1)
        mask=1.0-tf.cast(mask, dtype='float32')
        
        # Calculate Loss
        y_pred = tf.reshape(y_pred, (-1, 2, 3))                     # (batch, 2, 3)
        pred_coord =  tf.linalg.matmul(y_pred, self.stn_points)     # (batch, 2, 3) * (batch, 3, n_pionts) = (batch, 2, n_points)
        pred_coord = tf.transpose(pred_coord, perm=(0,2,1))
        pred_coord = tf.reshape(pred_coord, (-1, 8))
        losses = self.loss(y_true, pred_coord)                      # (batch, )
        
        losses = losses * mask
        losses_cnt = tf.reduce_sum(mask)
        
        # return tf.math.reduce_sum(losses) / losses_cnt
        return tf.math.reduce_sum(losses)

class SliceLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True
        )
    @tf.function(experimental_relax_shapes=True)
    def call(self, y_true, y_pred):
        # losses=(batch, 1, 200)
        losses=self.loss(y_true, y_pred)
        losses=tf.math.reduce_sum(losses, axis=(1,2))
        return tf.math.reduce_mean(losses)
