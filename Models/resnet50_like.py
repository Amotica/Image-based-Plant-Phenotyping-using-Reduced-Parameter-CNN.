from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
from keras.layers import *
from keras.layers import Input
from keras import layers
from keras.layers import Activation, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import ZeroPadding2D
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


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), padding="same", name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50_like(input_shape=(para.img_cols, para.img_rows, para.channels), classes=para.num_classes,
                  input_tensor=None):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=56,
                                      data_format=K.image_data_format(),
                                      include_top=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    #x = ZeroPadding2D((10, 10))(img_input)
    x = Conv2D(64, (7, 7), padding="same", name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), padding="same", strides=(2, 2))(x)
    y1 = x

    # Block 2
    x = conv_block(x, 3, [128, 128, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 256], stage=2, block='b')
    y2 = x


    # Block 3
    x = conv_block(x, 3, [256, 256, 512], stage=3, block='a')
    x = identity_block(x, 3, [256, 256, 512], stage=3, block='b')
    y3 = x

    # Block 4
    x = conv_block(x, 3, [512, 512, 1024], stage=4, block='a')
    x = identity_block(x, 3, [512, 512, 1024], stage=4, block='b')
    y4 = x

    # Block 5
    x = conv_block(x, 3, [1024, 1024, 2048], stage=5, block='a')
    x = identity_block(x, 3, [1024, 1024, 2048], stage=5, block='b')
    y5 = x


    #   =============
    #   The Decoder
    #   =============
    #   Block 5
    o = y5
    o = (Conv2D(classes, (1, 1), name='block5_conv1'))(o)

    #   Block 6
    o2 = y4
    o2 = (Conv2D(classes, (1, 1), name='block6_conv1'))(o2)
    o, o2 = crop(o, o2, img_input)
    o = Add()([o, o2])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 7
    o2 = y3
    o2 = (Conv2D(classes, (1, 1), name='block7_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 8
    o2 = y2
    o2 = (Conv2D(classes, (1, 1), name='block8_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 9
    o2 = y1
    o2 = (Conv2D(classes, (1, 1), name='block9_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    model = Model(img_input, o)
    cols = model.output_shape[1]
    rows = model.output_shape[2]

    o = Conv2D(classes, (1, 1))(o)
    o = Reshape((cols * rows, classes))(o)  # *****************
    o = Activation("softmax")(o)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, o, name='resnet38')

    return model, rows, cols


if __name__ == '__main__':
    m, rows, cols = ResNet50_like()
    print(m.summary())