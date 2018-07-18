from __future__ import absolute_import
from __future__ import print_function
from keras.layers import *
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import parameters as para
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.engine.topology import get_source_inputs


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


def fcn_8(input_shape=(para.img_cols, para.img_rows, para.channels), classes=para.num_classes, input_tensor=None):
    #img_input = Input(shape=input_shape)

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
    #print(img_input.shape)
    x = img_input
    # Encoder
    x = Conv2D(64, (3, 3), padding="same", name="block1_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same", name="block1_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    y1 = x

    x = Conv2D(128, (3, 3), padding="same", name="block2_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3, 3), padding="same", name="block2_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    y2 = x

    x = Conv2D(256, (3, 3), padding="same", name="block3_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same", name="block3_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    y3 = x

    x = Conv2D(512, (3, 3), padding="same", name="block4_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same", name="block4_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    y4 = x

    x = Conv2D(512, (3, 3), padding="same", name="block5_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same", name="block5_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    y5 = x


    #   =============
    #   The Decoder
    #   =============
    #   Block 6
    o = y5
    o = (Conv2D(classes, (1, 1), padding='same', name='block6_conv1'))(o)

    #   Block 7
    o2 = y4
    o2 = (Conv2D(classes, (1, 1), padding='same', name='block7_conv1'))(o2)
    o, o2 = crop(o, o2, img_input)
    o = Add()([o, o2])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 8
    o2 = y3
    o2 = (Conv2D(classes, (1, 1), padding='same', name='block8_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 9
    o2 = y2
    o2 = (Conv2D(classes, (1, 1), padding='same', name='block9_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)

    #   Block 10
    o2 = y1
    o2 = (Conv2D(classes, (1, 1), padding='same', name='block10_conv1'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])
    o = UpSampling2D(size=(2, 2))(o)


    model = Model(img_input, o)
    cols = model.output_shape[1]
    rows = model.output_shape[2]

    o = Conv2D(classes, (1, 1), padding="valid")(o)
    o = Reshape((cols*rows, classes))(o) #  *****************
    o = Activation("softmax")(o)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, o)
    return model, rows, cols


if __name__ == '__main__':
    m, rows, cols = fcn_8()
    print(m.summary())
    from keras.utils import plot_model
