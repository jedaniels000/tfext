"""

Extended functionality for TensorFlow including new layers, loss 
functions, and generators. Can be used directly, or just viewed for 
examples of implementing custom functionality.

TODO: Better document the 'only_non_null' option in losses/metrics
TODO: Increase abstraction with the 'only_non_null' option in 
    losses/metrics by allowing other null-valued inputs.
TODO: Have option to allow new best models to not replace old models
    in the CheckpointBest callback
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import h5py
import absl.logging
import wandb
from typing import List, Union

def train_test_split_indices(
    sample_ct: int, 
    train_size: float = 0.8, 
    seed: int = 7
) -> dict:
    """ 
    Create a training/testing split that can be used to index an 
    ndarray. 
    For example, if you have an ndarray of shape (100, 8, 8)
    named 'imgs' and want to split the 100 samples between a 
    training and testing set, you could use as follows:
    split_indices = train_test_split_indices(100)
    train_imgs = split_indices['train'] # 80 images (80, 8, 8)
    test_imgs = split_indices['test']   # 20 images (80, 8, 8)

    Parameters
    ----------
    img_ct: int
        the total number of samples; used to create indices
    train_size: float
        the percentage (range: (0.0, 1.0)) of data to use for the 
        training set
    seed: int
        the random seed to use in splitting the training and testing
        sets

    Returns
    -------
    dict
        the training and testing indices stored in a dictionary. Each
        are given as an np.ndarray and corresponding keys are 'train'
        and 'test', respectively.
    """

    # Set the random seed for reproducibility 
    np.random.seed(seed)
    # Get random index values
    train_choice = np.random.choice(
        range(sample_ct), size=(int(sample_ct * train_size),), replace=False
    )    
    # Create array to hold if value at index should be 
    # included in training
    train_ind = np.zeros(sample_ct, dtype=bool)
    train_ind[train_choice] = True
    # Take inverse for testing data
    test_ind = ~train_ind

    img_index = {
        'train': train_ind,
        'test': test_ind,
    }

    return img_index

def train_test_split(
    x: Union[np.ndarray, List[np.ndarray]], 
    y: np.ndarray, 
    train_size: float = 0.8,
    seed: int = 7
) -> tuple[dict, dict]:
    """
    Given input variable(s) 'x' and a target 'y', split
    between a training and testing dataset. Particularly
    useful with the added functionality to handle a list
    of input variables and split identically. Split along
    the first axis (axis 0).

    Parameters
    ----------
    x: Union[np.ndarray, List[np.ndarray]]
        the input samples. Can either be a single np.ndarray or a 
        list[np.ndarray] (in the case of multiple input variables)
    y: np.ndarray
        the target data
    train_size: float
        the percentage (range: (0.0, 1.0)) of data to use for the 
        training set
    seed: int
        the random seed to use in splitting the training and testing
        sets

    Returns
    -------
    tuple[dict, dict]
        tuple of dictionaries containing 2 keys each: 'train', for
        the training samples, and 'test', for the testing samples.
        If x was originally a list, then 'train' and 'test' will be
        a corresponding list of samples (np.ndarrays).
    """
    # Split into training and testing
    split_indices = train_test_split_indices(
        len(y), train_size=train_size, seed=seed
    )

    if isinstance(x, list):
        x = {
            'train': [var[split_indices['train']] for var in x],
            'test': [var[split_indices['test']] for var in x]
        }
    else:
        x = {
            'train': x[split_indices['train']],
            'test': x[split_indices['test']]
        }
    y = {
        'train': y[split_indices['train']],
        'test': y[split_indices['test']]
    }

    return x, y

class LayersExt:
    """ A conglomeration of additional Tensorflow layers """
    class SPConv2D(tf.keras.layers.Layer):
        """ Symmetrically Padded Conv2D """
        def __init__(
            self, 
            filters, 
            kernel_size, 
            strides=(1, 1), 
            activation='relu',
            **kwargs
        ) -> None:
            super(LayersExt.SPConv2D, self).__init__(**kwargs)
            self.filters = filters
            self.kernel_size = kernel_size
            self.strides = strides
            self.activation = activation

        def build(self, input_shape):
            n_dim = len(input_shape)
            # We determine the amount of padding needed based
            # on the kernel size. For example, we need a padding
            # of 1 in each direction for kernel_size=3, a padding
            # of 2 for kernel_size=5, etc.
            if isinstance(self.kernel_size, int):
                pad_size_x = int((self.kernel_size - 1) / 2)
                pad_size_y = pad_size_x
            else:
                pad_size_x = int((self.kernel_size[0] - 1) / 2)
                pad_size_y = int((self.kernel_size[1] - 1) / 2)
            # Padding to apply. We only apply padding along 
            # width axis (axis=-2) and height axis (axis=-3).
            # The channel axis (axis=-1) and all axes not
            # corresponding to width, height, or 
            # channel (axis=0:-3) are not padded as they should
            # not reduce in size from 2D convolution
            self.padding = tf.constant(
                [
                    *[[0, 0]]*(n_dim - 3), 
                    [pad_size_x, pad_size_x], 
                    [pad_size_y, pad_size_y], 
                    [0, 0]
                ]
            )

            self.conv = layers.Conv2D(
                self.filters, 
                self.kernel_size, 
                strides=self.strides, 
                activation=self.activation
            )

        def call(self, inputs):
            inputs = tf.pad(inputs, self.padding, mode='SYMMETRIC')
            x = self.conv(inputs)

            return x

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'activation': self.activation,
            })
            return config

    class SPConvLSTM2D(tf.keras.layers.Layer):
        """ Symmetrically Padded LSTM-Conv2D """
        def __init__(
            self, filters, kernel_size, 
            strides=(1, 1), activation=None, 
            return_sequences=False,
            **kwargs
        ) -> None:
            super(LayersExt.SPConvLSTM2D, self).__init__(**kwargs)
            self.filters = filters
            self.kernel_size = kernel_size
            self.strides = strides
            self.activation = activation
            self.return_sequences = return_sequences

        def build(self, input_shape):
            n_dim = len(input_shape)
            # We determine the amount of padding needed based
            # on the kernel size. For example, we need a padding
            # of 1 in each direction for kernel_size=3, a padding
            # of 2 for kernel_size=5, etc.
            if isinstance(self.kernel_size, int):
                pad_size_x = int((self.kernel_size - 1) / 2)
                pad_size_y = pad_size_x
            else:
                pad_size_x = int((self.kernel_size[0] - 1) / 2)
                pad_size_y = int((self.kernel_size[1] - 1) / 2)
            # Padding to apply. We only apply padding along 
            # width axis (axis=-2) and height axis (axis=-3).
            # The channel axis (axis=-1) and all axes not
            # corresponding to width, height, or 
            # channel (axis=0:-3) are not padded as they should
            # not reduce in size from 2D convolution
            self.padding = tf.constant(
                [
                    *[[0, 0]]*(n_dim - 3), 
                    [pad_size_x, pad_size_x], 
                    [pad_size_y, pad_size_y], 
                    [0, 0]
                ]
            )

            self.conv_lstm = layers.ConvLSTM2D(
                self.filters, 
                self.kernel_size, 
                strides=self.strides, 
                activation=self.activation,
                return_sequences=self.return_sequences
            )

        def call(self, inputs):
            inputs = tf.pad(inputs, self.padding, mode='SYMMETRIC')
            x = self.conv_lstm(inputs)

            return x
    
    class ChannelAttention(tf.keras.layers.Layer):
        """ Channel Attention Layer """
        # Ref: https://arxiv.org/pdf/1807.06521v2.pdf
        def __init__(
            self, reduction_ratio, **kwargs
        ) -> None:
            super(LayersExt.ChannelAttention, self).__init__(**kwargs)
            self.reduction_ratio = reduction_ratio

        def build(self, input_shape):
            self.max_pool = layers.GlobalAveragePooling2D(
                keepdims=True, name='max_pool'
            )
            self.avg_pool = layers.GlobalAveragePooling2D(
                keepdims=True, name='avg_pool'
            )
            self.mlp = tf.keras.Sequential(
                [
                    layers.Dense(
                        int(input_shape[-1] / self.reduction_ratio), 
                        activation='relu',
                        name='mlp_0'
                    ),
                    layers.Dense(input_shape[-1], name='mlp_1'),
                ]
            )
            self.sum_lyr = layers.Add()

        def call(self, inputs):
            max_pool = self.max_pool(inputs)
            avg_pool = self.avg_pool(inputs)

            mlp_max_pool = self.mlp(max_pool)
            mlp_avg_pool = self.mlp(avg_pool)

            summation = self.sum_lyr([mlp_max_pool, mlp_avg_pool])

            output = tf.keras.activations.sigmoid(summation)

            return output

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'reduction_ratio': self.reduction_ratio
            })
            return config

    class SpatialAttention(tf.keras.layers.Layer):
        """ Spatial Attention Layer """
        # Ref: https://arxiv.org/pdf/1807.06521v2.pdf
        def __init__(
            self, kernel_size=(3, 3), **kwargs
        ) -> None:
            super(LayersExt.SpatialAttention, self).__init__(**kwargs)
            self.kernel_size = kernel_size

        def build(self, input_shape):
            self.conv = LayersExt.SPConv2D(
                1, 
                self.kernel_size,
                activation='sigmoid',
            )

        def call(self, inputs):
            max_pool = tf.reduce_max(inputs, keepdims=True, axis=-1)
            avg_pool = tf.reduce_mean(inputs, keepdims=True, axis=-1)

            concat = layers.Concatenate()([max_pool, avg_pool])

            conv_layer = self.conv(concat)

            output = tf.keras.activations.sigmoid(conv_layer)

            return output

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'kernel_size': self.kernel_size
            })
            return config

    class CBAM(tf.keras.layers.Layer):
        """ Convolutional Block Attention Layer """
        # Ref: https://arxiv.org/pdf/1807.06521v2.pdf
        def __init__(
            self, reduction_ratio=1, kernel_size=3, 
            spatial_atten=True, **kwargs
        ) -> None:
            super(LayersExt.CBAM, self).__init__(**kwargs)
            self.reduction_ratio = reduction_ratio
            self.kernel_size = kernel_size
            self.spatial_attention=spatial_atten

        def build(self, input_shape):
            self.channel_attention = LayersExt.ChannelAttention(
                self.reduction_ratio
            )
            self.spatial_attention = LayersExt.SpatialAttention(
                self.kernel_size
            )

        def call(self, inputs):
            F = inputs
            attention = self.channel_attention(F)
            if self.spatial_attention:
                F = layers.Multiply()([attention, F])
                attention = self.spatial_attention(F)
            Fnn = layers.Multiply()([attention, F])

            return Fnn

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'reduction_ratio': self.reduction_ratio,
                'kernel_size': self.kernel_size
            })
            return config

    class NLCD_OHE(tf.keras.layers.Layer):
        """ One hot encoding layer for NLCD """

        def __init__(
            self, 
            # Whether or not to round to major classifications
            major=False, 
            # Whether or not the Tensor has a channel axis (which should be of
            # size 1)
            has_channel=False,
            **kwargs
        ) -> None:
            super(LayersExt.NLCD_OHE, self).__init__(**kwargs)

            # Legend for NLCD, where the first element is the label
            # of the classifiction provided by NLCD and the second
            # is the index we are assigning to that label
            self.NLCD_LEGEND = tf.constant([
                [11, 0],    # Water
                [12 ,0],    # Water
                [21, 1],    # Developed
                [22, 1],    # Developed
                [23, 1],    # Developed
                [24, 1],    # Developed
                [31, 2],    # Barren
                [41, 3],    # Forest
                [42, 3],    # Forest
                [43, 3],    # Forest
                [51, 4],    # Shrubland
                [52, 4],    # Shrubland
                [71, 5],    # Herbaceous
                [72, 5],    # Herbaceous
                [73, 5],    # Herbaceous
                [74, 5],    # Herbaceous
                [81, 6],    # Planted/Cultivated
                [82, 6],    # Planted/Cultivated
                [90, 7],    # Wetlands
                [95, 7]     # Wetlands
            ]) if major else tf.constant([
                [11, 0],    # Open Water
                [12 ,1],    # Perennial Ice/Snow
                [21, 2],    # Developed, Open Space
                [22, 3],    # Developed, Low Intensity
                [23, 4],    # Developed, Medium Intensity
                [24, 5],    # Developed, High Intensity
                [31, 6],    # Barren Land
                [41, 7],    # Deciduous Forest
                [42, 8],    # Evergreen Forest
                [43, 9],    # Mixed Forest
                [51, 10],   # Dwarf Scrub
                [52, 11],   # Shrub/Scrub
                [71, 12],   # Grassland/Herbaceous
                [72, 13],   # Sedge/Herbaceous
                [73, 14],   # Lichens
                [74, 15],   # Moss
                [81, 16],   # Pasture/Hay
                [82, 17],   # Cultivated Crops
                [90, 18],   # Woody Wetlands
                [95, 19]    # Emergent Herbaceous Wetlands
            ])

            self.NLCD_LOOKUP = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=self.NLCD_LEGEND[:, 0],
                    values=self.NLCD_LEGEND[:, 1],
                ),
                default_value=0,
                name="nlcd_legend"
            )
            self.has_channel = has_channel
            self.major = major

        def build(self, input_shape):
            self.cat_ct = 8 if self.major else 20

        def call(self, inputs):

            # If the Tensor has a channel axis, remove it
            if self.has_channel: inputs = inputs[...,0]

            # Indices specifying where there should be a true value (1)
            # in the encoded NLCD output
            true_indices = self.NLCD_LOOKUP.lookup(
                tf.cast(inputs, dtype=tf.dtypes.int32)
            )
            # Now that we have the indices, we can perform one-hot encoding
            # on the data
            return tf.one_hot(true_indices, depth=self.cat_ct)

        def get_config(self):
            return super(LayersExt.NLCD_OHE, self).get_config()

class LossesExt:
    """ A conglomeration of additional Tensorflow loss functions """
    @staticmethod
    def mse(only_non_null=False):
        """ MSE loss function """
        def mse(y_true, y_pred):
            if only_non_null:
                # Flatten arrays
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.reshape(y_true, [-1])

                # Get indices where actual values are greater than 0
                # (i.e. where there non null data)
                y_true_nonnull_indices = tf.reshape(tf.where(y_true > 0), [-1])

                y_pred = tf.gather(y_pred, indices=y_true_nonnull_indices)
                y_true = tf.gather(y_true, indices=y_true_nonnull_indices)
            return tf.reduce_mean((y_true - y_pred)**2)
        return mse

    @staticmethod
    def psnr(max_value):
        """ PSNR (Peak signal-to-noise ratio) loss function """
        def psnr(y_true, y_train):
            return -1 * tf.image.psnr(
                y_true, y_train, max_value, name=None
            )
        return psnr

    @staticmethod
    def ssim(max_value):
        """ SSIM (Structural Similarity Index) loss function """
        def ssim(y_true, y_train):
            return -1 * tf.reduce_mean(tf.image.ssim(
                y_true, y_train, max_value
            ))
        return ssim

class MetricsExt:
    """ A conglomeration of additional Tensorflow metrics """
    class NRMSE(tf.keras.metrics.Metric):
        def __init__(
            self, 
            name: str = 'nrmse', 
            only_non_null: bool = False, 
            **kwargs
        ) -> None:
            """
            Mean normalized root mean squared error Tensorflow metric

            Parameters
            ----------
            name: str
                the name to give to the metric
            only_non_null: bool
                find the location of the null values in the target data
                and only compute the metric on the corresponding 
                prediction values. Currently assumes that the null 
                values are those less than 0.

            Returns
            -------
            None
            """
            super(MetricsExt.NRMSE, self).__init__(name=name, **kwargs)
            self.sum_squared_error = self.add_weight(
                name='sse', initializer='zeros'
            )
            # Sum of true values (i.e. targets) for computing
            # the average of target values
            self.true_sum = self.add_weight(
                name='true_sum', initializer='zeros'
            )
            # The number of elements for which sum of squared errors
            # has been computed 
            self.count = self.add_weight(
                name='count', initializer='zeros'
            )
            self.only_non_null = only_non_null

        def update_state(self, y_true, y_pred, sample_weight=None):
            if self.only_non_null:
                # Flatten arrays
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.reshape(y_true, [-1])

                # Get indices where actual values are greater than 0
                # (i.e. where there non null data)
                y_true_nonnull_indices = tf.reshape(tf.where(y_true > 0), [-1])

                y_pred = tf.gather(y_pred, indices=y_true_nonnull_indices)
                y_true = tf.gather(y_true, indices=y_true_nonnull_indices)
                
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
            self.true_sum.assign_add(tf.reduce_sum(y_true))

            self.true_mean = self.true_sum / self.count

            values = tf.square(tf.subtract(y_true, y_pred))

            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                sample_weight = tf.broadcast_to(sample_weight, values.shape)
                values = tf.multiply(values, sample_weight)
            self.sum_squared_error.assign_add(tf.reduce_sum(values))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        def result(self):
            return tf.divide(
                tf.sqrt(tf.divide(self.sum_squared_error, self.count)), 
                self.true_mean
            )

    class RMSE(tf.keras.metrics.Metric):
        """ Root Mean Square error """
        def __init__(
            self, 
            name: str = 'rmse', 
            **kwargs
        ):
            """
            Root mean square error Tensorflow metric

            Parameters
            ----------
            name: str
                the name to give to the metric

            Returns
            -------
            None
            """

            super(MetricsExt.RMSE, self).__init__(name=name, **kwargs)
            self.sum_squared_error = self.add_weight(
                name='sse', initializer='zeros'
            )
            # The number of elements for which sum of squared errors
            # has been computed 
            self.count = self.add_weight(
                name='count', initializer='zeros'
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            values = tf.square(tf.subtract(y_true, y_pred))

            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                sample_weight = tf.broadcast_to(sample_weight, values.shape)
                values = tf.multiply(values, sample_weight)
            self.sum_squared_error.assign_add(tf.reduce_sum(values))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        def result(self):
            return tf.sqrt(tf.divide(self.sum_squared_error, self.count))

    class MSE(tf.keras.metrics.Metric):
        """ Mean Square error """
        def __init__(
            self, 
            name: str = 'mse', 
            only_non_null=False, 
            **kwargs
        ):
            """
            Mean squared error Tensorflow metric

            Parameters
            ----------
            name: str
                the name to give to the metric
            only_non_null: bool
                find the location of the null values in the target data
                and only compute the metric on the corresponding 
                prediction values. Currently assumes that the null 
                values are those less than 0.

            Returns
            -------
            None
            """

            super(MetricsExt.MSE, self).__init__(name=name, **kwargs)
            self.sum_squared_error = self.add_weight(
                name='sse', initializer='zeros'
            )
            # The number of elements for which sum of squared errors
            # has been computed 
            self.count = self.add_weight(
                name='count', initializer='zeros'
            )
            self.only_non_null = only_non_null

        def update_state(self, y_true, y_pred, sample_weight=None):
            if self.only_non_null:
                # Flatten arrays
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.reshape(y_true, [-1])

                # Get indices where actual values are greater than 0
                # (i.e. where there non null data)
                y_true_nonnull_indices = tf.reshape(tf.where(y_true > 0), [-1])

                y_pred = tf.gather(y_pred, indices=y_true_nonnull_indices)
                y_true = tf.gather(y_true, indices=y_true_nonnull_indices)

            values = tf.square(tf.subtract(y_true, y_pred))

            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                sample_weight = tf.broadcast_to(sample_weight, values.shape)
                values = tf.multiply(values, sample_weight)
            self.sum_squared_error.assign_add(tf.reduce_sum(values))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        def result(self):
            return tf.divide(self.sum_squared_error, self.count)

    class MAEE(tf.keras.metrics.Metric):
        """ Mean Absolute error """
        def __init__(
            self, 
            name: str = 'mae', 
            only_non_null=False, 
            **kwargs
        ):
            """
            Mean absolute error Tensorflow metric

            Parameters
            ----------
            name: str
                the name to give to the metric
            only_non_null: bool
                find the location of the null values in the target data
                and only compute the metric on the corresponding 
                prediction values. Currently assumes that the null 
                values are those less than 0.

            Returns
            -------
            None
            """

            super(MetricsExt.MAEE, self).__init__(name=name, **kwargs)
            self.sum_absolute_error = self.add_weight(
                name='sae', initializer='zeros'
            )
            # The number of elements for which sum of squared errors
            # has been computed 
            self.count = self.add_weight(
                name='count', initializer='zeros'
            )
            self.only_non_null = only_non_null

        def update_state(self, y_true, y_pred, sample_weight=None):
            if self.only_non_null:
                # Flatten arrays
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.reshape(y_true, [-1])

                # Get indices where actual values are greater than 0
                # (i.e. where there non null data)
                y_true_nonnull_indices = tf.reshape(tf.where(y_true > 0), [-1])

                y_pred = tf.gather(y_pred, indices=y_true_nonnull_indices)
                y_true = tf.gather(y_true, indices=y_true_nonnull_indices)

            values = tf.abs(tf.subtract(y_true, y_pred))

            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                sample_weight = tf.broadcast_to(sample_weight, values.shape)
                values = tf.multiply(values, sample_weight)
            self.sum_absolute_error.assign_add(tf.reduce_sum(values))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

        def result(self):
            return tf.divide(self.sum_absolute_error, self.count)

    class R2(tf.keras.metrics.Metric):
        def __init__(
            self, 
            name: str = 'r2', 
            only_non_null=False, 
            **kwargs
        ):
            """
            Coefficient of determination (R^2) Tensorflow metric

            Parameters
            ----------
            name: str
                the name to give to the metric
            only_non_null: bool
                find the location of the null values in the target data
                and only compute the metric on the corresponding 
                prediction values. Currently assumes that the null 
                values are those less than 0.

            Returns
            -------
            None
            """
            super(MetricsExt.R2, self).__init__(name=name, **kwargs)

            self.only_non_null = only_non_null

            # The number of elements for which the true sum has been
            # computed
            self.count = self.add_weight(
                name='count', initializer='zeros'
            )

            # Sum of true values (i.e. targets) for computing
            # the average of target values
            self.true_sum = self.add_weight(
                name='true_sum', initializer='zeros'
            )
            self.true_squared_summed = self.add_weight(
                name='true_squared_summed', initializer='zeros'
            )
            self.res_SS = self.add_weight(
                name='SSR', initializer='zeros'
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            if self.only_non_null:
                # Flatten arrays
                y_pred = tf.reshape(y_pred, [-1])
                y_true = tf.reshape(y_true, [-1])

                # Get indices where actual values are greater than 0
                # (i.e. where there non null data)
                y_true_nonnull_indices = tf.reshape(tf.where(y_true > 0), [-1])

                y_pred = tf.gather(y_pred, indices=y_true_nonnull_indices)
                y_true = tf.gather(y_true, indices=y_true_nonnull_indices)
            
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

            # Sum of Squared Errors
            self.true_sum.assign_add(tf.reduce_sum(y_true))

            self.true_squared_summed.assign_add(
                tf.reduce_sum(tf.square(y_true))
            )
            self.true_summed_squared = tf.square(self.true_sum)

            self.tot_SS = tf.subtract(
                self.true_squared_summed,
                tf.divide(self.true_summed_squared, self.count)
            )

            """ Sum of Squared Residuals"""
            res_vals = tf.square(tf.subtract(y_true, y_pred))

            self.res_SS.assign_add(tf.reduce_sum(res_vals))        

        def result(self):

            return tf.subtract(1.0, tf.math.divide(self.res_SS, self.tot_SS))

class GeneratorsExt:
    class H5Generator(tf.keras.utils.Sequence):  
        def __init__(
            self, 
            filepath: str, 
            train: bool,
            var_names: Union[str, list[str]],
            target_name: str,
            batch_size: int
        ) :
            """
            Tensorflow generator for use with H5 files.
            Assumes the data structure is first split
            into 'train' and 'test', then the name of the
            variable (e.g. /train/<var_name>), inside which 
            are the datasets for the variable.

            Parameters
            ----------
            filepath: str
                the path to the HDF5 file from which to pull data
            train: bool
                if True, will retrieve the training data (from /train) 
                rather than the testing data (from /test)
            var_names: Union[str, list[str]]
                String, or list of strings for multiple input 
                variables, that correspond to the names of the 
                input variables in the HDF5 file
            target_name: str
                the name of the target variable in the HDF5 file
            batch_size: int
                the size of the batches (in number of samples). If
                the number of samples is not perfectly divisible by
                the batch size, the last batch will just have the 
                remaining samples.

            Returns
            -------
            None
            """

            self.filepath = filepath
            # Convert the variable name variable to a list if not already
            # provided as list
            self.var_names = var_names if isinstance(
                var_names, list
            ) else [var_names]
            self.train_or_test = 'train' if train else 'test'

            self.target_name = target_name
            self.batch_size = batch_size
            
            # Get the number of images/samples based on the length of the
            # targets
            with h5py.File(self.filepath, 'r') as h5:
                # Get the keys for the target variables
                self.y_path = '/' + self.train_or_test + '/' + self.target_name
                y_group = h5[self.y_path]
                self.y_keys = np.array(
                    list(y_group.keys())
                )
                self.img_ct = len(
                    h5[self.y_path].keys()
                )

                # Get the keys for the input variables
                self.var_keys = []
                self.var_group_paths = []
                for var_name in self.var_names:
                    var_path = '/' + self.train_or_test + '/' + var_name
                    self.var_group_paths.append(var_path)
                    var_group = h5[var_path]
                    self.var_keys.append(np.array(
                        list(var_group.keys())
                    ))

            self.shuffled_indices = np.random.permutation(self.img_ct)
            self.n = 0
            self.max = self.__len__()
        
        def on_epoch_end(self):
            """ On epoch end, reshuffle keys (i.e. samples) """
            self.shuffled_indices = np.random.permutation(self.img_ct)

        def __len__(self) :
            # The number of batches per epoch
            return int(np.ceil(self.img_ct / self.batch_size))
        
        def __getitem__(self, index):
            # Indices of data which we want to extract for the current 
            # batch
            batch_start = index         * self.batch_size
            if (index + 1) * self.batch_size >= self.img_ct:
                batch_end = self.img_ct
            else:
                batch_end = (index + 1) * self.batch_size
            batch_indices = self.shuffled_indices[batch_start:batch_end]

            # Holds the x (model input) data
            batch_x = []

            # Get the specified samples using the x and y keys
            with h5py.File(self.filepath, 'r') as h5:
                for var_i, var_name in enumerate(self.var_names):
                    var_group = h5[self.var_group_paths[var_i]]
                    var_batch_keys = self.var_keys[var_i][batch_indices]
                    
                    batch_x.append(
                        np.stack([
                            var_group.get(key)[()] for key in var_batch_keys
                        ])
                    )

                y_group = h5[self.y_path]
                y_batch_keys = self.y_keys[batch_indices]
                batch_y = np.stack([
                    y_group.get(key)[()] for key in y_batch_keys
                ])

            # If the length of the x variables list is 1, meaning there 
            # is only a single input variable, convert to non-list by 
            # just extracting first element
            batch_x = batch_x[0] if len(batch_x) == 1 else batch_x

            return batch_x, batch_y

        def get_random_xy(self, count, seed=None):
            """ Get random input(s) and the matching target(s) """
            # Indices of data which we want to extract
            if seed is not None: np.random.seed(seed)
            random_indices = np.random.choice(self.img_ct, count)

            # Holds the x (model input) data
            batch_x = []

            with h5py.File(self.filepath, 'r') as h5:
                for var_i, var_name in enumerate(self.var_names):
                    var_group = h5[self.var_group_paths[var_i]]
                    var_batch_keys = self.var_keys[var_i][random_indices]
                    
                    batch_x.append(
                        np.stack([
                            var_group.get(key)[()] for key in var_batch_keys
                        ])
                    )

                y_group = h5[self.y_path]
                y_batch_keys = self.y_keys[random_indices]
                batch_y = np.stack([
                    y_group.get(key)[()] for key in y_batch_keys
                ])

            # If the length of the x variables list is 1, meaning there is
            # only a single input variable, convert to non-list by just
            # extracting first element
            batch_x = batch_x[0] if len(batch_x) == 1 else batch_x

            return batch_x, batch_y

        def __next__(self):
            if self.n >= self.max:
                self.n = 0
            result = self.__getitem__(self.n)
            self.n += 1
            return result

class CallbacksExt:
    """ A conglomeration of additional Tensorflow callbacks """
    class CheckpointBest(tf.keras.callbacks.Callback):
        def __init__(
            self, 
            filepath: str = 'saved_model', 
            monitor: str ='loss',
            # Percent difference between training and validation
            # sets allowed when saving. This ensures that the model
            # doesn't overfit to the validation set with a great 
            # validation loss but poor training loss. If none, this
            # is ignored and the best model for validation is saved.
            tolerance_percent: float = None,
            verbose: bool = True
        ) -> None:
            """
            Checkpoint (save) the best performing model with respect
            to the validation metric provided. Each time a new best
            is found, it will replace the previous saved model.

            Parameters
            ----------
            filepath: str
                the filepath at which to save the checkpointed model
            monitor: str
                the metric to monitor for determining the best model
            tolerance_percent: float
                percent difference between training and validation sets
                allowed when saving. This ensures that the model 
                doesn't overfit to the validation set with a great 
                validation loss but poor training loss. If None, this
                is ignored and the best model for validation is saved.
            verbose: bool
                if True, outputs the previous best validation loss, the
                new validation loss, and whether or not the new model
                will be saved (i.e. when the new validation loss is 
                better than the previous best)

            Returns
            -------
            None
            """
            super(CallbacksExt.CheckpointBest, self).__init__()
            self.monitor = monitor
            self.filepath = filepath
            self.tolerance_percent = tolerance_percent
            self.verbose = verbose
        def on_train_begin(self, logs=None):
            # Initialize the best as infinity.
            self.best_loss = np.Inf
            absl.logging.set_verbosity(absl.logging.ERROR)
        def on_epoch_end(self, epoch, logs=None):
            train_loss = logs.get(self.monitor)
            val_loss = logs.get("val_" + self.monitor)

            # If tolerance percent is provided, and if the training loss is 
            # greater than the validation loss (meaning that the validation 
            # set is possibly being overfit to), check if the percent 
            # difference is within the threshold
            if (self.tolerance_percent is not None) and (train_loss>val_loss):
                percent_diff = abs(
                    ((train_loss - val_loss) * 2) / (train_loss + val_loss)
                )
                tolerance_met = percent_diff < self.tolerance_percent
            else:
                tolerance_met = True

            # If the percent difference between training loss and validation 
            # loss is less than specified tolerance and the validation loss 
            # is less than the best loss, save the model
            new_best_loss = np.less(val_loss, self.best_loss)
            old_best_loss = self.best_loss
            if tolerance_met and new_best_loss:
                self.best_loss = val_loss
                self.model.save(self.filepath)

            if self.verbose:
                print()
                if not(tolerance_met): print("Tolerance not met. Not saving.")
                if new_best_loss: print(
                    f"New best validation loss!"
                )
                print(f"Previous best validation loss: {round(old_best_loss, 2)}")
                print(f"Current validation loss: {round(val_loss, 2)}")
                print()


    class LogSampleWandb(tf.keras.callbacks.Callback):
        def __init__(
            self,
            validation_generator: GeneratorsExt.H5Generator,
            sample_ct: int = 1,
            log_target: bool = True
        ) -> None:
            """
            Log a sample of the prediction output to Wandb. Only
            applicable to image data.

            Parameters
            ----------
            validation_generator: GeneratorsExt.H5Generator
                the generator from which to get the input to the model
                whose prediction is logged to Wandb
            sample_ct: int
                the number of samples to log
            log_target: bool
                if True, will log the target image (in addition to
                the prediction)

            Returns
            -------
            None
            """

            super(CallbacksExt.LogSampleWandb, self).__init__()
            self.sample_ct = sample_ct
            self.validation_generator = validation_generator
            self.log_target = log_target
        def on_epoch_end(self, epoch, logs=None):
            x, y = self.validation_generator.get_random_xy(self.sample_ct)

            predict = self.model.predict(x)

            if self.log_target:
                # Concatenate the prediction and target into a single
                # array so they can be viewed side-by-side
                imgs = np.concatenate([predict, y], axis=-2)
            else:
                imgs = predict
            # Normalize the image values so they are easier to 
            # visualize
            img_min = imgs.min()
            imgs = imgs - img_min
            img_max = imgs.max()
            imgs = imgs/img_max

            imgs = [wandb.Image(img) for img in imgs]

            wandb.log({
                "predictions": imgs,
            })