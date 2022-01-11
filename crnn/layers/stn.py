# [ref] https://github.com/oarriaga/paz/blob/master/examples/spatial_transfomer_networks/layers.py
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf


class BilinearInterpolation(Layer):
    """Performs bilinear interpolation as a keras layer
    # References
        [1]  Spatial Transformer Networks, Max Jaderberg, et al.
        [2]  https://github.com/skaae/transformer_network
        [3]  https://github.com/EderSantana/seya
        [4]  https://github.com/oarriaga/STN.keras
    """

    def __init__(self, output_size, dynamic=True, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(dynamic=dynamic, **kwargs)

    def get_config(self):
        return {'output_size': self.output_size, 'name': self.name}
        
    def build(self, input_shapes):
        if len(input_shapes) == 2:
            self.input_channel=input_shapes[0][-1]
        else:
            self.input_channel=input_shapes[-1]
        super(BilinearInterpolation, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        '''
        input_shapes: list, list of two tensor shape
            tensors[0], tuple, (None, h, w, 3), shape of images
            tenosrs[1], tuple, (None, 6), shape of transformation mats
        '''
        height = self.output_size[0]
        width = self.output_size[1]
        return (None, height, width, input_shapes[0][-1])

    @tf.function(experimental_relax_shapes=True)
    def call(self, tensors, mask=None):
        '''
        tensors: list, list of two tensors
            tensors[0], tf.Tensor=(None, h, w, 3), batch of image
            tenosrs[1], tf.Tenosr=(None, 6), batch of transformation mat
        '''
        heihgt = self.output_size[0]
        width = self.output_size[1]
        image, affine_transforms = tensors
        batch_size, num_channels = tf.shape(image)[0], tf.shape(image)[3]
        affine_transforms = K.reshape(affine_transforms, (batch_size, 2, 3))
        grids = self._make_a_grid_per_batch(heihgt, width, batch_size)
        # Transform Coordinate
        grids = K.batch_dot(affine_transforms, grids)
        # Resampling Image
        interpolated_image = self._interpolate(image, grids, self.output_size)
        new_shape = (batch_size, heihgt, width, self.input_channel)
        interpolated_image = tf.reshape(interpolated_image, new_shape)
        return interpolated_image

    def _make_grid(self, height, width):
        # Make Grid Coordinate in range [-1, 1]
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)
        return grid

    def _make_a_grid_per_batch(self, height, width, batch_size):
        grid = self._make_grid(height, width)
        grid = K.flatten(grid)                                      # (3*w*h)
        grids = K.tile(grid, K.stack([batch_size]))                 # (batch, 3*w*h)
        grids = K.reshape(grids, (batch_size, 3, height * width))   # (batch, 3, w*h)
        return grids

    def _interpolate(self, image, grids, output_size):
        img_shape = K.shape(image)
        batch_size = img_shape[0]
        height = img_shape[1]
        width = img_shape[2]
        num_channels = img_shape[3]
        x = K.cast(K.flatten(grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(grids[:, 1:2, :]), dtype='float32')
        # convert coordinate from [-1, 1] to [0, 1]
        x, y = self._to_image_coordinates(x, y, (height, width))
        # compute each coordinate's two bounding integer coordinate
        x_min, y_min, x_max, y_max = self._compute_corners(x, y)
        # clip coordinate to 1, no larger than 1
        x_min, y_min = self._clip_to_valid_coordinates((x_min, y_min), image)
        x_max, y_max = self._clip_to_valid_coordinates((x_max, y_max), image)
        
        # calcuate each image's start index
        offsets = self._compute_offsets_for_flat_batch(image, output_size)
        indices = self._calculate_indices(offsets, (x_min, y_min), (x_max, y_max), width)
        flat_images = K.reshape(image, shape=(-1, num_channels))
        flat_images = K.cast(flat_images, dtype='float32')
        pixel_values = self._gather_pixel_values(flat_images, indices)
        x_min, y_min = self._cast_points_to_float((x_min, y_min))
        x_max, y_max = self._cast_points_to_float((x_max, y_max))
        areas = self._calculate_areas(x, y, (x_min, y_min), (x_max, y_max))
        return self._compute_interpolations(areas, pixel_values)

    def _to_image_coordinates(self, x, y, shape):
        x = (0.5 * (x + 1.0)) * K.cast(shape[1], dtype='float32')
        y = (0.5 * (y + 1.0)) * K.cast(shape[0], dtype='float32')
        return x, y

    def _compute_corners(self, x, y):
        x_min, y_min = K.cast(x, 'int32'), K.cast(y, 'int32')
        x_max, y_max = x_min + 1, y_min + 1
        return x_min, y_min, x_max, y_max

    def _clip_to_valid_coordinates(self, points, image):
        x, y = points
        max_y = tf.shape(image)[1]-1 #K.int_shape(image)[1] - 1
        max_x = tf.shape(image)[2]-1 #K.int_shape(image)[2] - 1
        x = K.clip(x, 0, max_x)
        y = K.clip(y, 0, max_y)
        return x, y

    def _compute_offsets_for_flat_batch(self, image, output_size):
        img_shape = K.shape(image)[0:3]
        batch_size = img_shape[0]
        height = img_shape[1]
        width = img_shape[2]
        coordinates_per_batch = K.arange(0, batch_size) * (height * width)      # (batch,)
        coordinates_per_batch = K.expand_dims(coordinates_per_batch, axis=-1)   # (batch, 1)
        flat_output_size = output_size[0] * output_size[1]
        coordinates_per_batch_per_pixel = K.repeat_elements(                    # (batch, w*h)
            coordinates_per_batch, flat_output_size, axis=1)
        coordinates_per_batch_per_pixel = K.flatten(coordinates_per_batch_per_pixel)
        return coordinates_per_batch_per_pixel                                  # (batch*w*h, )

    def _calculate_indices(self, base, top_left_corners, bottom_right_corners, width):
        (x_min, y_min), (x_max, y_max) = top_left_corners, bottom_right_corners
        y_min_offset = base + (y_min * width)
        y_max_offset = base + (y_max * width)
        indices_top_left = y_min_offset + x_min
        indices_top_right = y_max_offset + x_min
        indices_low_left = y_min_offset + x_max
        indices_low_right = y_max_offset + x_max
        return (indices_top_left, indices_top_right,
                indices_low_left, indices_low_right)

    def _gather_pixel_values(self, flat_image, indices):
        pixel_values_A = K.gather(flat_image, indices[0])
        pixel_values_B = K.gather(flat_image, indices[1])
        pixel_values_C = K.gather(flat_image, indices[2])
        pixel_values_D = K.gather(flat_image, indices[3])
        return (pixel_values_A, pixel_values_B, pixel_values_C, pixel_values_D)

    def _calculate_areas(self, x, y, top_left_corners, bottom_right_corners):
        (x_min, y_min), (x_max, y_max) = top_left_corners, bottom_right_corners
        area_A = K.expand_dims(((x_max - x) * (y_max - y)), 1) # right bottom area
        area_B = K.expand_dims(((x_max - x) * (y - y_min)), 1) # right top    area
        area_C = K.expand_dims(((x - x_min) * (y_max - y)), 1) # left  bottom area
        area_D = K.expand_dims(((x - x_min) * (y - y_min)), 1) # left  top    area
        return area_A, area_B, area_C, area_D
        
    def _cast_points_to_float(self, points):
        return K.cast(points[0], 'float32'), K.cast(points[1], 'float32')

    def _compute_interpolations(self, areas, pixel_values):
        weighted_area_A = pixel_values[0] * areas[0]
        weighted_area_B = pixel_values[1] * areas[1]
        weighted_area_C = pixel_values[2] * areas[2]
        weighted_area_D = pixel_values[3] * areas[3]
        interpolation = (weighted_area_A + weighted_area_B +
                         weighted_area_C + weighted_area_D)
        return interpolation