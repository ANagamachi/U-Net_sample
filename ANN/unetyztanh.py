#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout

class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE_X = 96
        self.INPUT_IMAGE_SIZE_Y = 64
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # (X x Y x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE_X, self.INPUT_IMAGE_SIZE_Y, input_channel_count))

        # エンコーダーの作成
        # (X/2 x Y/2 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (X/4 x Y/4 x 2N)
        filter_count = first_layer_filter_count*2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (X/8 x Y/8 x 4N)
        filter_count = first_layer_filter_count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (X/16 x Y/16 x 8N)
        filter_count = first_layer_filter_count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (X/32 x Y/32 x 8N)
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (X/64 x Y/64 x 8N)
#        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (2 x 2 x 8N)
#        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (1 x 1 x 8N)
#        enc8 = self._add_encoding_layer(filter_count, enc7)

        # デコーダーの作成
        # (2 x 2 x 8N)
#        dec1 = self._add_decoding_layer(filter_count, False, enc8)
#        dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
#        dec2 = self._add_decoding_layer(filter_count, False, dec1)
#        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (2 x 2 x 8N)
#        dec1 = self._add_decoding_layer(filter_count, False, enc6)
#        dec1 = concatenate([dec1, enc5], axis=self.CONCATENATE_AXIS)

        # (X/4 x Y/4 x 8N)
        dec1 = self._add_decoding_layer(filter_count, False, enc5)
        dec1 = concatenate([dec1, enc4], axis=self.CONCATENATE_AXIS)

        # (X/8 x Y/8 x 4N)
        filter_count = first_layer_filter_count*4
        dec2 = self._add_decoding_layer(filter_count, False, dec1)
        dec2 = concatenate([dec2, enc3], axis=self.CONCATENATE_AXIS)

        # (X/4 x Y/4 x 2N)
        filter_count = first_layer_filter_count*2
        dec3 = self._add_decoding_layer(filter_count, False, dec2)
        dec3 = concatenate([dec3, enc2], axis=self.CONCATENATE_AXIS)

        # (X/2 x Y/2 x N)
        filter_count = first_layer_filter_count
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = concatenate([dec4, enc1], axis=self.CONCATENATE_AXIS)

        # (X x Y x output_channel_count)
        dec5 = Activation(activation='tanh')(dec4)
        dec5 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec5)
        dec5 = Activation(activation='linear')(dec5)

        self.UNET = Model(input=inputs, output=dec5)

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = Activation(activation='tanh')(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='tanh')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET
