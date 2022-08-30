import tensorflow as tf
import numpy as np
from tensorflow import keras
from decoders import CTCDecoder
from losses import diou_loss, iou


class SequenceAccuracy(keras.metrics.Metric):
    def __init__(self, name='sequence_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    @tf.function(experimental_relax_shapes=True)
    def sparse2dense(self, tensor, shape):
        tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor = tf.cast(tensor, tf.float32)
        return tensor

    # @tf.function(experimental_relax_shapes=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # (batch_size, max_label_size)
        y_true_shape = tf.shape(y_true)
        batch_size = y_true_shape[0]
        # (batch_size, timestep, classes)
        y_pred_shape = tf.shape(y_pred)
        max_width = tf.math.maximum(y_true_shape[1], y_pred_shape[1])
        logit_length = tf.fill([batch_size], y_pred_shape[1])      
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            top_paths=5, 
            sequence_length=logit_length)

        # (batch, timestep)
        y_true = self.sparse2dense(y_true, [batch_size, max_width])
        # (batch, timestep)
        y_pred = self.sparse2dense(decoded[0], [batch_size, max_width])
        error_list = tf.math.reduce_any(
            tf.math.not_equal(y_true, y_pred), axis=1)
        error_list = tf.cast(error_list, tf.float32)
        num_errors = tf.math.reduce_sum(error_list)
        batch_size = tf.cast(batch_size, tf.float32)
        
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)
        # Return List of Correct Case(1 is correct, 0 is wrong)
        return 1.0 - error_list

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)
    
    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)


class SequenceTop5Accuracy(keras.metrics.Metric):
    def __init__(self, name='sequence_accuracy', table_path='example/digit.txt',**kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, 
            tf.string, tf.lookup.TextFileIndex.WHOLE_LINE), '')
        
    def detokenize(self, x):
        x = tf.RaggedTensor.from_sparse(x)
        x = tf.ragged.map_flat_values(self.table.lookup, x)
        strings = tf.strings.reduce_join(x, axis=1)
        return strings
    
    @tf.function(experimental_relax_shapes=True)
    def sparse2dense(self, tensor, shape):
        tensor = tf.sparse.reset_shape(tensor, shape)
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor = tf.cast(tensor, tf.float32)
        return tensor

    # @tf.function(experimental_relax_shapes=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # (batch_size, max_label_size)
        y_true_shape = tf.shape(y_true)
        batch_size = y_true_shape[0]
        # (batch_size, timestep, classes)
        y_pred_shape = tf.shape(y_pred)
        max_width = tf.math.maximum(y_true_shape[1], y_pred_shape[1])
        logit_length = tf.fill([batch_size], y_pred_shape[1])      
        decoded, _ = tf.nn.ctc_beam_search_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            top_paths=5,
            sequence_length=logit_length)
        
        # # (batch, timestep)
        # y_true = self.sparse2dense(y_true, [batch_size, max_width])
        # # (batch, timestep)
        # y_pred = self.sparse2dense(decoded[0], [batch_size, max_width])
        
        # error_list = tf.math.reduce_any(
        #     tf.math.not_equal(y_true, y_pred), axis=1)
        # error_list = tf.cast(error_list, tf.float32)
        # num_errors = tf.math.reduce_sum(error_list)
        any_correct=False
        single_truth = self.detokenize(y_true)
        for i in range(5):
            single_pred  = self.detokenize(decoded[i])
            print('single_truth', single_truth, 'single_pred', single_pred)
            if single_pred==single_truth:
                any_correct=True
                break
        
            
        num_errors = 0 if any_correct else 1
        batch_size = tf.cast(batch_size, tf.float32)
        
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)
        # Return List of Correct Case(1 is correct, 0 is wrong)
        return tf.constant([1.0]) - num_errors

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)
    
    def reset_state(self):
        self.count.assign(0)
        self.total.assign(0)


class BoxAccuracy(keras.metrics.Metric):
    def __init__(self, name='box_accuracy', iou_threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.iou = self.add_weight(name="iou", initializer='zeros')

        self.iou_threshold = iou_threshold
        self.stn_points = tf.constant([[ 
            [-1.0, -1.0, 1.0], # Left  Top
            [ 1.0,  1.0, 1.0], # Right Bottom
            [ 1.0, -1.0, 1.0], # Right Top
            [-1.0,  1.0, 1.0], # Left  Bottom
        ]])
        self.stn_points = tf.transpose(self.stn_points, perm=(0,2,1)) # (1, 3, n_points)
    @tf.function(experimental_relax_shapes=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_true: (batch, 2, 4)
        y_pred: (batch, 6)
        """
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype='float32')
        y_pred = tf.reshape(y_pred, (-1, 2, 3))                     # (batch, 2, 3)
        pred_coord = tf.linalg.matmul(y_pred, self.stn_points)    # (batch, 2, 3) * (batch, 3, n_pionts) = (batch, 2, n_points)
        pred_coord = tf.transpose(pred_coord, perm=(0,2,1))
        pred_coord = tf.reshape(pred_coord, (-1, 8))
        iou_val = iou(y_true, pred_coord)

        correct_list=tf.cast(iou_val >= self.iou_threshold, dtype='float32')
        num_correct = tf.reduce_sum(correct_list)

        self.total.assign_add(batch_size)
        self.count.assign_add(num_correct)
        self.iou.assign_add(tf.reduce_sum(iou_val))
        # Return List of Correct Case(1 is correct, 0 is wrong)
        return correct_list

    def result(self):
        return self.count / self.total

    def result_iou(self):
        return self.iou / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)
        self.iou.assign(0)


class EditDistance(keras.metrics.Metric):
    def __init__(self, name='edit_distance', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sum_distance = self.add_weight(name='sum_distance', 
                                            initializer='zeros')
                
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        logit_length = tf.fill([batch_size], y_pred_shape[1])      
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        sum_distance = tf.math.reduce_sum(tf.edit_distance(decoded[0], y_true))
        batch_size = tf.cast(batch_size, tf.float32)
        self.sum_distance.assign_add(sum_distance)
        self.total.assign_add(batch_size)

    def result(self):
        return self.sum_distance / self.total

    def reset_states(self):
        self.sum_distance.assign(0)
        self.total.assign(0)