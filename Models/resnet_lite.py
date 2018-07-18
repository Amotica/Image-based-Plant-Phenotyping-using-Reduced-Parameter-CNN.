from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import *
from keras.layers import Activation, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, UpSampling2D, SeparableConvolution2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import parameters as para
from keras.regularizers import l2

# crop the images so that they will be the same size for adding or merging
def crop(out_1, out_2, input):
    o_shape2 = Model(input, out_2).output_shape
    output_rows2 = o_shape2[1]
    output_cols2 = o_shape2[2]

    o_shape1 = Model(input, out_1).output_shape
    output_rows1 = o_shape1[1]
    output_cols1 = o_shape1[2]

    abs_x = abs(output_cols1 - output_cols2)
    abs_y = abs(output_rows2 - output_rows1)

    if output_cols1 > output_cols2:
        out_1 = Cropping2D(cropping=((0, 0), (0, abs_x)))(out_1)
    else:
        out_2 = Cropping2D(cropping=((0, 0), (0, abs_x)))(out_2)

    if output_rows1 > output_rows2:
        out_1 = Cropping2D(cropping=((0, abs_y), (0, 0)))(out_1)
    else:
        out_2 = Cropping2D(cropping=((0, abs_y), (0, 0)))(out_2)

    return out_1, out_2


def _conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation_rate=(1, 1), alpha=1.0):
    #   filter array
    filters1, filters2 = filters

    # determine the channel axis
    if K.image_data_format() == 'channels_last':
        ch_axis = 3
        row_axis = 1
        col_axis = 2
    else:
        ch_axis = 1
        row_axis = 2
        col_axis = 3

    #   set the name of the convolution + batch normalisation + padding
    conv_name_base = 'Group_' + str(stage) + '_Conv_' + str(block)
    bn_name_base = 'Group_' + str(stage) + '_bn_' + str(block)
    relu_name_base = 'Group_' + str(stage) + '_relu_' + str(block)

    # Block 0
    x = Conv2D(int(filters1 * alpha), (1, 1), name=conv_name_base + '_0a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConvolution2D(int(filters1 * alpha), kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate, name=conv_name_base + '_0')(x)
    x = BatchNormalization(name=bn_name_base + '_0')(x)
    x = Activation('relu', name=relu_name_base + '_0')(x)
    x = Conv2D(int(filters1 * alpha), (1, 1), name=conv_name_base + '_0b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 1
    x = Conv2D(int(filters2 * alpha), (1, 1), name=conv_name_base + '_1a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConvolution2D(int(filters2 * alpha), kernel_size, padding='same', dilation_rate=dilation_rate, name=conv_name_base + '_1')(x)
    x = BatchNormalization(name=bn_name_base + '_1')(x)
    x = Activation('relu', name=relu_name_base + '_1')(x)
    x = Conv2D(int(filters2 * alpha), (1, 1), name=conv_name_base + '_1b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #   determine the input shape and residual shape and use this to compute strides for the shortcut conv2D
    input_shape = K.int_shape(input_tensor) #   input shape
    x_shape = K.int_shape(x)    #   residual shape
    stride_width = int(round(input_shape[row_axis] / x_shape[row_axis]))
    stride_height = int(round(input_shape[col_axis] / x_shape[col_axis]))
    equal_channels = input_shape[ch_axis] == x_shape[ch_axis]
    #   the shortcut will be the same as the input tensor, if and only if they are of the same shape,
    # otherwise reshape the shortcut using convolution
    shortcut = input_tensor
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        #print('reshaping shortcut using convolution...')
        shortcut = Conv2D(filters=x_shape[ch_axis],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="same",
                          name=conv_name_base + '_2')(input_tensor)
        shortcut = BatchNormalization(axis=ch_axis, name=bn_name_base + '_2')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu', name=relu_name_base + '_2')(x)
    return x


#   Basic Block is used for ResNet 18-Layers and 34-Layers
def _identity_block(input_tensor, kernel_size, filters, stage, block, alpha=1):
    #   filter array
    filters1, filters2 = filters

    #   set the name of the convolution + batch normalisation + padding
    conv_name_base = 'Group_' + str(stage) + '_Conv_' + str(block)
    bn_name_base = 'Group_' + str(stage) + '_bn_' + str(block)
    relu_name_base = 'Group_' + str(stage) + '_relu_' + str(block)

    x = Conv2D(int(filters1 * alpha), (1, 1), name=conv_name_base + '_0a')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConvolution2D(int(filters1 * alpha), kernel_size, padding='same', name=conv_name_base + '_0')(x)
    x = BatchNormalization(name=bn_name_base + '_0')(x)
    x = Activation('relu', name=relu_name_base + '_0')(x)
    x = Conv2D(int(filters1 * alpha), (1, 1), name=conv_name_base + '_0b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(int(filters2 * alpha), (1, 1), name=conv_name_base + '_1a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConvolution2D(int(filters2 * alpha), kernel_size, padding='same', name=conv_name_base + '_1')(x)
    x = BatchNormalization(name=bn_name_base + '_1')(x)
    x = Conv2D(int(filters2 * alpha), (1, 1), name=conv_name_base + '_1b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu', name=relu_name_base + '_1')(x)

    return x


def build_resnet_blocks(input_tensor, kernel_size, filters, group=1, strides=(2, 2), _identities=2, alpha=1.0):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = input_tensor

    conv_basename = "res_Group_" + str(group) + "_Conv_"
    bn_basename = "res_Group_" + str(group) + "_bn_"
    relu_basename = "res_Group_" + str(group) + "_relu_"
    maxpool_basename = "res_Group_" + str(group) + "_maxpool_"

    if group == 1:
        x = Conv2D(filters[0], kernel_size, strides=strides, padding='same', name=conv_basename + '0')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_basename + '0')(x)
        x = Activation('relu', name=relu_basename + '0')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=maxpool_basename + '0')(x)
    else:
        x = _conv_block(x, kernel_size, filters, stage=group, block=0, strides=strides, alpha=alpha)
        for block in range(_identities):
            x = _identity_block(x, kernel_size, filters, stage=group, block=block+1, alpha=alpha)
    return x


def ResNet18(include_top=False, input_tensor=None, input_shape=(para.img_rows, para.img_cols, para.channels), alpha=1.0):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    #   =======
    #   Group 1
    #   =======
    x = build_resnet_blocks(img_input, kernel_size=(7, 7), filters=[64], group=1, strides=(2, 2), _identities=0, alpha=alpha)
    y1 = x

    #   =======
    #   Group 2
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[64, 64], group=2, strides=(1, 1), _identities=1, alpha=alpha)
    y2 = x

    #   =======
    #   Group 3
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[128, 128], group=3, strides=(2, 2), _identities=1, alpha=alpha)
    y3 = x

    #   =======
    #   Group 4
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[256, 256], group=4, strides=(2, 2), _identities=1, alpha=alpha)
    y4 = x

    #   =======
    #   Group 5
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[512, 512], group=5, strides=(2, 2), _identities=1, alpha=alpha)
    y5 = x

    #   =============
    #   The Decoder
    #   =============
    #   Block 6
    o = y5
    o = (Conv2D(para.num_classes, (1, 1), padding='same', name='block6_conv1'))(o)

    #   Block 7
    o2 = y4
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block7_conv1'))(o2)
    o, o2 = crop(o, o2, img_input)
    o = Add()([o, o2])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 8
    o2 = y3
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block8_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 9
    o2 = y2
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block9_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 10
    o2 = y1
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block10_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(4, 4))(o)

    model = Model(img_input, o)
    cols = model.output_shape[1]
    rows = model.output_shape[2]

    o = Conv2D(para.num_classes, (1, 1), padding="valid")(o)
    o = Reshape((cols * rows, para.num_classes))(o)  # *****************
    o = Activation("softmax")(o)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, o)
    return model, rows, cols


def ResNet34(include_top=False, input_tensor=None, input_shape=(para.img_rows, para.img_cols, para.channels), alpha=1.0):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    #   =======
    #   Group 1
    #   =======
    x = build_resnet_blocks(img_input, kernel_size=(7, 7), filters=[64], group=1, strides=(2, 2), _identities=0, alpha=alpha)
    y1 = x

    #   =======
    #   Group 2
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[64, 64], group=2, strides=(1, 1), _identities=2, alpha=alpha)
    y2 = x

    #   =======
    #   Group 3
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[128, 128], group=3, strides=(2, 2), _identities=3, alpha=alpha)
    y3 = x

    #   =======
    #   Group 4
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[256, 256], group=4, strides=(2, 2), _identities=5, alpha=alpha)
    y4 = x

    #   =======
    #   Group 5
    #   =======
    x = build_resnet_blocks(x, kernel_size=(3, 3), filters=[512, 512], group=5, strides=(2, 2), _identities=2, alpha=alpha)
    y5 = x

    #   =============
    #   The Decoder
    #   =============
    #   Block 6
    o = y5
    o = (Conv2D(para.num_classes, (1, 1), padding='same', name='block6_conv1'))(o)

    #   Block 7
    o2 = y4
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block7_conv1'))(o2)
    o, o2 = crop(o, o2, img_input)
    o = Add()([o, o2])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 8
    o2 = y3
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block8_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 9
    o2 = y2
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block9_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 10
    o2 = y1
    o2 = (Conv2D(para.num_classes, (1, 1), padding='same', name='block10_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(4, 4))(o)

    model = Model(img_input, o)
    cols = model.output_shape[1]
    rows = model.output_shape[2]

    o = Conv2D(para.num_classes, (1, 1), padding="valid")(o)
    o = Reshape((cols * rows, para.num_classes))(o)  # *****************
    o = Activation("softmax")(o)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, o)
    return model, rows, cols


if __name__ == '__main__':
    m, rows, cols  = ResNet34(alpha=1.0)
    print(m.summary())