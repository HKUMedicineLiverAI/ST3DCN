from keras.layers import *
from keras import backend as K
import keras
import tensorflow as tf
from keras.models import Model
import numpy as np


def attention_module_2d(net, attention_module):
    if attention_module == 'se_block':  # SE_Block
        net = se_block_2d(net)
    elif attention_module == 'cbam_block':
        net = cbam_block_2d(net)
    elif attention_module == 'eca_block':
        net = eca_block_2d(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block_2d(input_feature, ratio=8):
    # suppose input_features: (B, 128, 128, 32], --> channels = 32
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature._keras_shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)    # (B, H, W, C] ---> (B, C) where C: channels
    se_feature = Reshape((1, 1, channel))(se_feature)       # (B, C) --> (B, 1, 1, C]
    assert se_feature._keras_shape[1:] == (1, 1, channel)

    se_feature = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel//ratio)

    se_feature = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)

    se_feature = multiply([input_feature, se_feature])   # (B, H, W, C) * (B, 1, 1, C) ---> (B, H, W, C)
    print(se_feature)
    return se_feature


def cbam_block_2d(cbam_feature, ratio=8):
    # cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = eca_block_2d(cbam_feature)
    cbam_feature = spatial_attention_2d(cbam_feature)
    return cbam_feature


def eca_block_2d(input_feature):
    import keras
    avg_pool = GlobalAveragePooling2D()(input_feature)    # [B, H, W, Channels] --> [B, channels]
    unsqueezed_avg_pool = Lambda(lambda x: keras.backend.expand_dims(x, 1))(avg_pool)   # ---> [B, 1, channels]
    permuted_avg_pool = Permute((2, 1))(unsqueezed_avg_pool)     # ---> [B, channels, 1]

    # The output of Conv1D is [B, channels, 1]
    conv_avg_pool = Conv1D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(permuted_avg_pool)
    permuted_conv_output = Permute((2, 1))(conv_avg_pool)  # ---> [B, 1, channels]

    unsqueezed = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(permuted_conv_output)  # ---> [B, 1, 1, channels]
    output_feature = multiply([input_feature, unsqueezed])   # --> Output: [B, H, W, Channels], e.g., (B, 128, 128, 32]

    return output_feature


def spatial_attention_2d(input_feature):
    import keras
    kernel_size = 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    avg_pool = Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    max_gool = Lambda(lambda x: K.max(x, axis=channel_axis, keepdims=True))(input_feature)
    concat = keras.layers.Concatenate(axis=channel_axis)([avg_pool, max_gool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)
    # input_feature: (B, H, W, C) * (B, H, W, 1) ===> (B, H, W, C)

    return multiply([input_feature, cbam_feature])


def attention_module_3d(net, attention_module):
    if attention_module == 'se_block':  # SE_Block
        net = se_block_3d(net)
    elif attention_module == 'cbam_block':
        net = cbam_block_3d(net)
    elif attention_module == 'eca_block':
        net = eca_block_3d(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block_3d(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature._keras_shape[channel_axis]
    # print(channel_axis, channel)
    se_feature = GlobalAveragePooling3D()(input_feature)
    se_feature = Reshape((1, 1, 1, channel))(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, 1, channel)

    se_feature = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, 1, channel//ratio)

    se_feature = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, 1, channel)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block_3d(cbam_feature, ratio=8):
    # cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = eca_block_3d(cbam_feature)
    cbam_feature = spatial_attention_3d(cbam_feature)
    return cbam_feature


def channel_attention_3d(input_feature, ratio=4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, 1, channel)

    max_pool = GlobalMaxPooling3D()(input_feature)
    max_pool = Reshape((1, 1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, 1, channel)

    cbam_feature = keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention_3d(input_feature):
    kernel_size = 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    avg_pool = Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    max_gool = Lambda(lambda x: K.max(x, axis=channel_axis, keepdims=True))(input_feature)
    concat = keras.layers.Concatenate(axis=channel_axis)([avg_pool, max_gool])
    cbam_feature = Conv3D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


def eca_block_3d(input_feature):
    import keras
    avg_pool = GlobalAveragePooling3D()(input_feature)    # [B, H, W, D, Channels] --> [B, channels]
    unsqueezed_avg_pool = Lambda(lambda x: keras.backend.expand_dims(x, 1))(avg_pool)   # ---> [B, 1, channels]
    permuted_avg_pool = Permute((2, 1))(unsqueezed_avg_pool)     # ---> [B, channels, 1]
    # squeezed_avg_pool_w = Lambda(lambda x: keras.backend.squeeze(x, 1))(squeezed_avg_pool_h)  # ---> [B, 1, channels]

    # The output of Conv1D is [B, channels, 1]
    conv_avg_pool = Conv1D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(permuted_avg_pool)
    permuted_conv_output = Permute((2, 1))(conv_avg_pool)  # ---> [B, channels, 1]

    unsqueezed_w = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(permuted_conv_output)  # ---> [B, 1, channels, 1]

    unsqueezed_h = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(unsqueezed_w)    # ---> [B, 1, 1, channels, 1]

    output_feature = multiply([input_feature, unsqueezed_h])
    return output_feature


def reshape_3d_to_2d(inputs, depth):
    def get_slice(x, h1, h2):
        return x[:, :, :, h1:h2, :]

    # [B, H, W, D, 1] ---> [B*D, H, W, 1, 1]
    for i in range(depth):
        if i == 0:  # The first slice
            inputs_2d = Lambda(lambda x: get_slice(x, h1=0, h2=1))(inputs)
        else:
            inputs_slices = Lambda(lambda x: get_slice(x, h1=i, h2=i+1))(inputs)
            inputs_2d = concatenate([inputs_2d, inputs_slices], axis=0)

    # [B*D, H, W, 1, 1]  ---> [B*D, H, W, 1]  after squeeze operation
    inputs_2d = Lambda(lambda x: keras.backend.squeeze(x, axis=-1))(inputs_2d)
    return inputs_2d


def restore_2d_to_3d(inputs, depth, batch_size):
    for i in range(depth):
        if i == 0:
            inputs_retored = Lambda(lambda x: x[i:i+batch_size])(inputs)
        else:
            outputs_slice = Lambda(lambda x: x[i*batch_size: i*batch_size+batch_size])(inputs)
            inputs_retored = concatenate([inputs_retored, outputs_slice], axis=-1)

    return inputs_retored


def reshape_3d_to_p3d(inputs, depth):
    def get_slice(x, h1, h2):
        return x[:, :, :, h1:h2, :]

    # [B, H, W, D, 1] ---> [B*s, H, W, 4, 1], where = int((depth-slice_num)/slice_stride) + 1
    slice_num = 4
    slice_stride = 2
    num_batch = int((depth-slice_num)/slice_stride) + 1
    for i in range(0, num_batch):
        if i == 0:  # The first slice
            inputs_p3d = Lambda(lambda x: get_slice(x, h1=0, h2=4))(inputs)
        else:
            inputs_slices = Lambda(lambda x: get_slice(x, h1=i*2, h2=i*2+4))(inputs)
            inputs_p3d = concatenate([inputs_p3d, inputs_slices], axis=0)

    # [B*s, H, W, 1, 1]  ---> [B*D, H, W, 1]  after squeeze operation ---> (B*s, 128, 128, 4, 1)
    import keras
    inputs_p3d = Lambda(lambda x: keras.backend.squeeze(x, axis=-1))(inputs_p3d)     # (B*s, 128, 128, 4)
    return inputs_p3d


def restore_p3d_to_3d(inputs, depth, batch_size):
    overlap_slice_num = 2      # this depends on the setting in reshape_3d_to_p3d
    for i in range(depth):
        if i == 0:
            inputs_retored = Lambda(lambda x: x[i:i+batch_size])(inputs)
        else:
            outputs_slice = Lambda(lambda x: x[i*batch_size+overlap_slice_num: i*batch_size+batch_size])(inputs)
            inputs_retored = concatenate([inputs_retored, outputs_slice], axis=-1)

    return inputs_retored


def multi_scale_get_model_DCN(width=128, height=128, depth=128, batch_size=2, num_class=3, factor=2, num_gpu=2):
    inputs_A_resized = Input(shape=(width, height, depth, 1))
    '''
    import tensorflow as tf
    print('*1', tf.math.reduce_max(inputs_A_resized), tf.math.reduce_min(inputs_A_resized), tf.math.reduce_mean(inputs_A_resized),tf.math.reduce_std(inputs_A_resized))
    import keras
    inputs_A_sque = Lambda(lambda x: keras.backend.squeeze(x, axis=-1))(inputs_A_resized)
    inputs_A_sque_resize = Lambda(lambda x: tf.image.resize(x,[128,128], preserve_aspect_ratio=True))(inputs_A_sque)
    inputs_A = Lambda(lambda x: keras.backend.expand_dims(x, -1))(inputs_A_sque_resize)

    inputs_A_to_2d = reshape_3d_to_2d(inputs_A, depth=depth)     # (B, 128, 128, 128, 1) ---> (B*128, 128, 128, 1)
    inputs_A_to_p3d = reshape_3d_to_p3d(inputs_A, depth=depth)
    '''
    #print(inputs_A.shape, inputs_A_resized.shape, "Line-285")
    inputs_A_resized_2d = reshape_3d_to_2d(inputs_A_resized, depth=depth)

    inputs_A_resized_p3d = reshape_3d_to_p3d(inputs_A_resized, depth=depth)

    tf.print(inputs_A_resized.shape,inputs_A_resized_p3d.shape, batch_size, "Line-252")

    # The model of the encoding path of the first branch for (B, 256, 256, 128, 1): 3D inputs
    # (B, 128, 128, 128, 1) --> (B, 128, 128, 128, 16) --> (B, 64, 64, 64, 16)
    x1_3d_conv1_1 = Conv3D(filters=8*factor, kernel_size=3, padding='same', strides=1)(inputs_A_resized)
    x1_3d_ac1_1 = Activation('relu')(x1_3d_conv1_1)
    x1_a1_3d_1 = attention_module_3d(x1_3d_ac1_1, 'cbam_block')
    x1_3d_add1_1 = keras.layers.add([x1_3d_ac1_1, x1_a1_3d_1])
    x1_3d_bn1_1 = BatchNormalization()(x1_3d_add1_1)
    x1_3d_conv1_2 = Conv3D(filters=8*factor, kernel_size=3, padding='same', strides=1)(x1_3d_bn1_1)
    x1_3d_ac1_2 = Activation('relu')(x1_3d_conv1_2)
    x1_a1_3d_2 = attention_module_3d(x1_3d_ac1_2, 'cbam_block')
    x1_3d_add1_2 = keras.layers.add([x1_3d_ac1_2, x1_a1_3d_2])
    x1_3d_bn1_2 = BatchNormalization()(x1_3d_add1_2)
    x1_3d_max1 = MaxPool3D(pool_size=(2, 2, 2))(x1_3d_bn1_2)
    #tf.print('*2', tf.math.reduce_max(x1_3d_max1), tf.math.reduce_min(x1_3d_max1), tf.math.reduce_mean(x1_3d_max1),tf.math.reduce_std(x1_3d_max1))

    # (B, 64, 64, 64, 16) --> (B, 64, 64, 64, 32) --> (B, 32, 32, 32, 32)
    x1_3d_conv2_1 = Conv3D(filters=16*factor, kernel_size=3, padding='same', strides=1)(x1_3d_max1)
    x1_3d_ac2_1 = Activation('relu')(x1_3d_conv2_1)
    x1_a2_3d_1 = attention_module_3d(x1_3d_ac2_1, 'cbam_block')
    x1_3d_add2_1 = keras.layers.add([x1_3d_ac2_1, x1_a2_3d_1])
    x1_3d_bn2_1 = BatchNormalization()(x1_3d_add2_1)
    x1_3d_conv2_1 = Conv3D(filters=16*factor, kernel_size=3, padding='same', strides=1)(x1_3d_bn2_1)
    x1_3d_ac2_2 = Activation('relu')(x1_3d_conv2_1)
    x1_a2_3d_2 = attention_module_3d(x1_3d_ac2_2, 'cbam_block')
    x1_3d_add2_2 = keras.layers.add([x1_3d_ac2_2, x1_a2_3d_2])
    x1_3d_bn2_2 = BatchNormalization()(x1_3d_add2_2)
    x1_3d_max2 = MaxPool3D(pool_size=(2, 2, 2))(x1_3d_bn2_2)
    #tf.print('*3', tf.math.reduce_max(x1_3d_max2), tf.math.reduce_min(x1_3d_max2), tf.math.reduce_mean(x1_3d_max2),tf.math.reduce_std(x1_3d_max2))

    # (B, 32, 32, 32, 32) -> (B, 32, 32, 32, 64) --> (B, 16, 16, 16, 64)
    x1_3d_conv3_1 = Conv3D(filters=32*factor, kernel_size=3, padding='same', strides=1)(x1_3d_max2)
    x1_3d_ac3_1 = Activation('relu')(x1_3d_conv3_1)
    x1_a3_3d_1 = attention_module_3d(x1_3d_ac3_1, 'cbam_block')
    x1_3d_add3_1 = keras.layers.add([x1_3d_ac3_1, x1_a3_3d_1])
    x1_3d_bn3_1 = BatchNormalization()(x1_3d_add3_1)
    x1_3d_conv3_2 = Conv3D(filters=32*factor, kernel_size=3, padding='same', strides=1)(x1_3d_bn3_1)
    x1_3d_ac3_2 = Activation('relu')(x1_3d_conv3_2)
    x1_a3_3d_2 = attention_module_3d(x1_3d_ac3_2, 'cbam_block')
    x1_3d_add3_2 = keras.layers.add([x1_3d_ac3_2, x1_a3_3d_2])
    x1_3d_bn3_2 = BatchNormalization()(x1_3d_add3_2)
    x1_3d_max3 = MaxPool3D(pool_size=(2, 2, 2))(x1_3d_bn3_2)
    #tf.print('*4', tf.math.reduce_max(x1_3d_max3), tf.math.reduce_min(x1_3d_max3), tf.math.reduce_mean(x1_3d_max3),tf.math.reduce_std(x1_3d_max3))

    # (B, 16, 16, 16, 64) --> (B, 16, 16, 16, 128) ---> (B, 8, 8, 8, 128) -- GAP--> x1_3d_gap (B, 128)
    x1_3d_conv4_1 = Conv3D(filters=64*factor, kernel_size=3, padding='same', strides=1)(x1_3d_max3)
    x1_3d_ac4_1 = Activation('relu')(x1_3d_conv4_1)
    x1_a4_3d_1 = attention_module_3d(x1_3d_ac4_1, 'cbam_block')
    x1_3d_add4_1 = keras.layers.add([x1_3d_ac4_1, x1_a4_3d_1])
    x1_3d_bn4_1 = BatchNormalization()(x1_3d_add4_1)
    x1_3d_conv4_2 = Conv3D(filters=64*factor, kernel_size=3, padding='same', strides=1)(x1_3d_bn4_1)
    x1_3d_ac4_2 = Activation('relu')(x1_3d_conv4_2)
    x1_a4_3d_2 = attention_module_3d(x1_3d_ac4_2, 'cbam_block')
    x1_3d_add4_2 = keras.layers.add([x1_3d_ac4_2, x1_a4_3d_2])
    x1_3d_bn4_2 = BatchNormalization()(x1_3d_add4_2)           # (B, 16, 16, 16, 128)
    x1_3d_max4 = MaxPool3D(pool_size=(2, 2, 2))(x1_3d_bn4_2)
    x1_3d_gap = GlobalAveragePooling3D()(x1_3d_max4)   # Here is the output of the first path with original input as inputs
    #tf.print('*5', tf.math.reduce_max(x1_3d_bn4_2), tf.math.reduce_min(x1_3d_bn4_2), tf.math.reduce_mean(x1_3d_bn4_2),tf.math.reduce_std(x1_3d_bn4_2))
    #tf.print('*5.1', tf.math.reduce_max(x1_3d_gap), tf.math.reduce_min(x1_3d_gap), tf.math.reduce_mean(x1_3d_gap),tf.math.reduce_std(x1_3d_gap))
    """
    conc4 = concatenate([x1_3d_max3, x1_3d_bn4_2], axis=-1)   # (B, 16, 16, 16, 192)

    up4 = concatenate([Conv3DTranspose(32 * factor, (2, 2, 2), strides=(2, 2, 2), padding='same')
                       (conc4), x1_3d_bn3_2], axis=-1)   # (B, 16, 16, 16, 192) --> (B, 32, 32, 32, 64) (B, 32, 32, 32, 64) --> (B, 32, 32, 32, 128)
    conv4 = Conv3D(32 * factor, (3, 3, 3), padding='same')(up4)    # (B, 32, 32, 32, 64)
    conv4_ac = Activation('relu')(conv4)
    conv4_att = attention_module_3d(conv4_ac, 'cbam_block')
    conv4_add = keras.layers.add([conv4_ac, conv4_att])
    conv4_bn = BatchNormalization()(conv4_add)           # (B, 32, 32, 32, 64)
    tf.print('*6', tf.math.reduce_max(conv4_bn), tf.math.reduce_min(conv4_bn), tf.math.reduce_mean(conv4_bn),tf.math.reduce_std(conv4_bn))

    conc5 = concatenate([up4, conv4_bn], axis=-1)        # (B, 32, 32, 192)
    up5 = concatenate([Conv3DTranspose(16 * factor, (2, 2, 2), strides=(2, 2, 2), padding='same')
                       (conc5), x1_3d_bn2_2], axis=-1)   # (B, 32, 32, 32, 192) --> (B, 64, 64, 64, 32) (B, 64, 64, 64, 32) --> (B, 64, 64, 64, 64)
    conv5 = Conv3D(16 * factor, (3, 3, 3), padding='same')(up5)    # (B, 64, 64, 64, 32)
    conv5_ac = Activation('relu')(conv5)
    conv5_att = attention_module_3d(conv5_ac, 'cbam_block')
    conv5_add = keras.layers.add([conv5_ac, conv5_att])
    conv5_bn = BatchNormalization()(conv5_add)           # (B, 64, 64, 64, 32)
    tf.print('*7', tf.math.reduce_max(conv5_bn), tf.math.reduce_min(conv5_bn), tf.math.reduce_mean(conv5_bn),tf.math.reduce_std(conv5_bn))

    conc6 = concatenate([up5, conv5_bn], axis=-1)                 # (B, 64, 64, 64, 96)
    # (B, 64, 64, 64, 96) --> (B, 128, 128, 128, 16)  + (B, 128, 128, 128, 16) --> (B, 128, 128, 128, 32)
    up6 = concatenate([Conv3DTranspose(8 * factor, (2, 2, 2), strides=(2, 2, 2), padding='same')
                       (conc6), x1_3d_bn1_2], axis=-1)
    conv6 = Conv3D(8 * factor, (3, 3, 3), padding='same')(up6)    # (B, 128, 128, 128, 16)
    conv6_ac = Activation('relu')(conv6)
    conv6_att = attention_module_3d(conv6_ac, 'cbam_block')
    conv6_add = keras.layers.add([conv6_ac, conv6_att])
    conv6_bn = BatchNormalization()(conv6_add)                    # (B, 128, 128, 128, 16)
    tf.print('*8', tf.math.reduce_max(conv6_bn), tf.math.reduce_min(conv6_bn), tf.math.reduce_mean(conv6_bn),tf.math.reduce_std(conv6_bn))

    conv_1st_3d_output = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6_bn)  # (B, 128, 128, 128, 1)


    # The encoding path for the first branch with 2D inputs:
    x1_2d = Conv2D(filters=8*factor, kernel_size=3, strides=1)(inputs_A_to_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a1_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a1_2d])
    x1_2d = BatchNormalization()(x1_2d)
    x1_2d = Conv2D(filters=8*factor, kernel_size=3, strides=1)(x1_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a1_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a1_2d])
    x1_2d = BatchNormalization()(x1_2d)
    x1_2d = MaxPool2D(pool_size=(2, 2))(x1_2d)

    x1_2d = Conv2D(filters=16*factor, kernel_size=3, strides=1)(x1_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a2_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a2_2d])
    x1_2d = BatchNormalization()(x1_2d)
    x1_2d = Conv2D(filters=16*factor, kernel_size=3, strides=1)(x1_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a2_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a2_2d])
    x1_2d = BatchNormalization()(x1_2d)
    x1_2d = MaxPool2D(pool_size=(2, 2))(x1_2d)

    x1_2d = Conv2D(filters=32*factor, kernel_size=3, strides=1)(x1_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a3_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a3_2d])
    x1_2d = BatchNormalization()(x1_2d)
    x1_2d = Conv2D(filters=32*factor, kernel_size=3, strides=1)(x1_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a3_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a3_2d])
    x1_2d = BatchNormalization()(x1_2d)
    x1_2d = MaxPool2D(pool_size=(2, 2))(x1_2d)

    x1_2d = Conv2D(filters=64*factor, kernel_size=3, strides=1)(x1_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a4_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a4_2d])
    x1_2d = BatchNormalization()(x1_2d)
    x1_2d = Conv2D(filters=64*factor, kernel_size=3, strides=1)(x1_2d)
    x1_2d = Activation('relu')(x1_2d)
    x1_a4_2d = attention_module_2d(x1_2d, 'cbam_block')
    x1_2d = keras.layers.add([x1_2d, x1_a4_2d])
    x1_2d = BatchNormalization()(x1_2d)
    # x1_2d = layers.MaxPool2D(pool_size=(2, 2))(x1_2d)
    x1_2d_gap = GlobalAveragePooling2D()(x1_2d)
    """

    # The encoding path for the first branch with P3D inputs:
    # inputs_A_to_p3d: (B, 128, 128, 4) --> (B, 128, 128, 32)
    '''
    x1_p3d = Conv2D(filters=8*factor, kernel_size=3, strides=1)(inputs_A_to_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a1_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a1_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    x1_p3d = Conv2D(filters=8*factor, kernel_size=3, strides=1)(x1_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a1_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a1_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    x1_p3d = MaxPool2D(pool_size=(2, 2))(x1_p3d)
    tf.print('*9', tf.math.reduce_max(x1_p3d), tf.math.reduce_min(x1_p3d), tf.math.reduce_mean(x1_p3d),tf.math.reduce_std(x1_p3d))

    x1_p3d = Conv2D(filters=16*factor, kernel_size=3, strides=1,)(x1_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a2_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a2_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    x1_p3d = Conv2D(filters=16*factor, kernel_size=3, strides=1,)(x1_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a2_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a2_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    x1_p3d = MaxPool2D(pool_size=(2, 2))(x1_p3d)
    tf.print('*10', tf.math.reduce_max(x1_p3d), tf.math.reduce_min(x1_p3d), tf.math.reduce_mean(x1_p3d),tf.math.reduce_std(x1_p3d))

    x1_p3d = Conv2D(filters=32*factor, kernel_size=3, strides=1,)(x1_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a3_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a3_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    x1_p3d = Conv2D(filters=32*factor, kernel_size=3, strides=1)(x1_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a3_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a3_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    x1_p3d = MaxPool2D(pool_size=(2, 2))(x1_p3d)
    tf.print('*11', tf.math.reduce_max(x1_p3d), tf.math.reduce_min(x1_p3d), tf.math.reduce_mean(x1_p3d),tf.math.reduce_std(x1_p3d))

    x1_p3d = Conv2D(filters=64*factor, kernel_size=3, strides=1)(x1_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a4_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a4_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    x1_p3d = Conv2D(filters=64*factor, kernel_size=3, strides=1)(x1_p3d)
    x1_p3d = Activation('relu')(x1_p3d)
    x1_a4_p3d = attention_module_2d(x1_p3d, 'cbam_block')
    x1_p3d = keras.layers.add([x1_p3d, x1_a4_p3d])
    x1_p3d = BatchNormalization()(x1_p3d)
    # x1_p3d = MaxPool2D(pool_size=(2, 2))(x1_p3d)
    x1_p3d_gap = GlobalAveragePooling2D()(x1_p3d)
    tf.print('*12', tf.math.reduce_max(x1_p3d_gap), tf.math.reduce_min(x1_p3d_gap), tf.math.reduce_mean(x1_p3d_gap),tf.math.reduce_std(x1_p3d_gap))
    '''
    # Here resized_input with the shape of (B, 256, 256, 128, 1)
    '''
    x2_3d_conv1_1 = Conv3D(filters=8 * factor, kernel_size=3, strides=1, padding='same')(inputs_A_resized)
    x2_3d_act1_1 = Activation('relu')(x2_3d_conv1_1)
    x2_a1_3d_1 = attention_module_3d(x2_3d_act1_1, 'cbam_block')
    x2_3d_add1_1 = keras.layers.add([x2_3d_act1_1, x2_a1_3d_1])
    x2_3d_bn1_1 = BatchNormalization()(x2_3d_add1_1)
    x2_3d_conv1_2 = Conv3D(filters=8 * factor, kernel_size=3, strides=1, padding='same')(x2_3d_bn1_1)
    x2_3d_act1_2 = Activation('relu')(x2_3d_conv1_2)
    x2_a1_3d_2 = attention_module_3d(x2_3d_act1_2, 'cbam_block')
    x2_3d_add1_2 = keras.layers.add([x2_3d_act1_2, x2_a1_3d_2])
    x2_3d_bn1_2 = BatchNormalization()(x2_3d_add1_2)
    x2_3d_max1 = MaxPool3D(pool_size=(2, 2, 2))(x2_3d_bn1_2)  # change from 3 to 2
    #tf.print('*13', tf.math.reduce_max(x2_3d_max1), tf.math.reduce_min(x2_3d_max1), tf.math.reduce_mean(x2_3d_max1),tf.math.reduce_std(x2_3d_max1))

    x2_3d_conv2_1 = Conv3D(filters=16 * factor, kernel_size=3, strides=1, padding='same')(x2_3d_max1)
    x2_3d_act2_1 = Activation('relu')(x2_3d_conv2_1)
    x2_a2_3d_1 = attention_module_3d(x2_3d_act2_1, 'cbam_block')
    x2_3d_add2_1 = keras.layers.add([x2_3d_act2_1, x2_a2_3d_1])
    x2_3d_bn2_1 = BatchNormalization()(x2_3d_add2_1)
    x2_3d_conv2_1 = Conv3D(filters=16 * factor, kernel_size=3, strides=1, padding='same')(x2_3d_bn2_1)
    x2_3d_act2_2 = Activation('relu')(x2_3d_conv2_1)
    x2_a2_3d_2 = attention_module_3d(x2_3d_act2_2, 'cbam_block')
    x2_3d_add2_1 = keras.layers.add([x2_3d_act2_2, x2_a2_3d_2])
    x2_3d_bn2_2 = BatchNormalization()(x2_3d_add2_1)
    x2_3d_max2 = MaxPool3D(pool_size=(2, 2, 2))(x2_3d_bn2_2)
    #tf.print('*14', tf.math.reduce_max(x2_3d_max2), tf.math.reduce_min(x2_3d_max2), tf.math.reduce_mean(x2_3d_max2),tf.math.reduce_std(x2_3d_max2))

    x2_3d_conv3_1 = Conv3D(filters=32 * factor, kernel_size=3, strides=1, padding='same')(x2_3d_max2)
    x2_3d_act3_1 = Activation('relu')(x2_3d_conv3_1)
    x2_a3_3d_1 = attention_module_3d(x2_3d_act3_1, 'cbam_block')
    x2_3d_add3_1 = keras.layers.add([x2_3d_act3_1, x2_a3_3d_1])
    x2_3d_bn3_1 = BatchNormalization()(x2_3d_add3_1)
    x2_3d_conv3_2 = Conv3D(filters=32 * factor, kernel_size=3, strides=1, padding='same')(x2_3d_bn3_1)
    x2_3d_act3_2 = Activation('relu')(x2_3d_conv3_2)
    x2_a3_3d_2 = attention_module_3d(x2_3d_act3_2, 'cbam_block')
    x2_3d_add3_2 = keras.layers.add([x2_3d_act3_2, x2_a3_3d_2])
    x2_3d_bn3_2 = BatchNormalization()(x2_3d_add3_2)
    x2_3d_max3 = MaxPool3D(pool_size=(2, 2, 2))(x2_3d_bn3_2)
    #tf.print('*15', tf.math.reduce_max(x2_3d_max3), tf.math.reduce_min(x2_3d_max3), tf.math.reduce_mean(x2_3d_max3),tf.math.reduce_std(x2_3d_max3))

    x2_3d_conv4_1 = Conv3D(filters=64 * factor, kernel_size=3, strides=1, padding='same')(x2_3d_max3)
    x2_3d_act4_1 = Activation('relu')(x2_3d_conv4_1)
    x2_a4_3d_1 = attention_module_3d(x2_3d_act4_1, 'cbam_block')
    x2_3d_add4_1 = keras.layers.add([x2_3d_act4_1, x2_a4_3d_1])
    x2_3d_bn4_1 = BatchNormalization()(x2_3d_add4_1)
    x2_3d_conv4_2 = Conv3D(filters=64 * factor, kernel_size=3, strides=1, padding='same')(x2_3d_bn4_1)
    x2_3d_act4_2 = Activation('relu')(x2_3d_conv4_2)
    x2_a4_3d_2 = attention_module_3d(x2_3d_act4_2, 'cbam_block')
    x2_3d_add4_2 = keras.layers.add([x2_3d_act4_2, x2_a4_3d_2])
    x2_3d_bn4_2 = BatchNormalization()(x2_3d_add4_2)
    # x2_3d_max4 = MaxPool3D(pool_size=(3, 3, 3))(x2_3d_bn4_2)
    x2_3d_gap = GlobalAveragePooling3D()(x2_3d_bn4_2)     # Here is the output of the first path with original input as inputs
    #tf.print('*16', tf.math.reduce_max(x2_3d_bn4_2), tf.math.reduce_min(x2_3d_bn4_2), tf.math.reduce_mean(x2_3d_bn4_2),tf.math.reduce_std(x2_3d_bn4_2))
    #tf.print('*16.1', tf.math.reduce_max(x2_3d_gap), tf.math.reduce_min(x2_3d_gap), tf.math.reduce_mean(x2_3d_gap),tf.math.reduce_std(x2_3d_gap))

    # (B, 32, 32, 16, 64) + (B, 32, 32, 16, 128) ---> (B, 32, 32, 16, 192)
    conc4_2 = concatenate([x2_3d_max3, x2_3d_bn4_2], axis=-1)
    up4_2 = concatenate([Conv3DTranspose(32 * factor, (2, 2, 2), strides=(2, 2, 2), padding='same')
                         (conc4_2), x2_3d_bn3_2], axis=-1)      # (B, 32, 32, 16, 192) --> (B, 64, 64, 32, 64) (B, 64, 64, 32, 64) --> (B, 64, 64, 32, 128)
    conv4_2 = Conv3D(32 * factor, (3, 3, 3), padding='same')(up4_2)    # (B, 64, 64, 32, 64)
    conv4_ac_2 = Activation('relu')(conv4_2)
    conv4_att_2 = attention_module_3d(conv4_ac_2, 'cbam_block')
    conv4_add_2 = keras.layers.add([conv4_ac_2, conv4_att_2])
    conv4_bn_2 = BatchNormalization()(conv4_add_2)                     # (B, 64, 64, 32, 64)
    tf.print('*17', tf.math.reduce_max(conv4_bn_2), tf.math.reduce_min(conv4_bn_2), tf.math.reduce_mean(conv4_bn_2),tf.math.reduce_std(conv4_bn_2))

    conc5_2 = concatenate([up4_2, conv4_bn_2], axis=-1)                # (B, 64, 64, 32, 192)
    up5_2 = concatenate([Conv3DTranspose(16 * factor, (2, 2, 2), strides=(2, 2, 2), padding='same')
                       (conc5_2), x2_3d_bn2_2], axis=-1)   # (B, 64, 64, 32, 192) --> (B, 128, 128, 64, 32) (B, 128, 128, 64, 32) --> (B, 128, 128, 64, 64)
    conv5_2 = Conv3D(16 * factor, (3, 3, 3), padding='same')(up5_2)    # (B, 128, 128, 64, 32)
    conv5_ac_2 = Activation('relu')(conv5_2)
    conv5_att_2 = attention_module_3d(conv5_ac_2, 'cbam_block')
    conv5_add_2 = keras.layers.add([conv5_ac_2, conv5_att_2])
    conv5_bn_2 = BatchNormalization()(conv5_add_2)                     # (B, 128, 128, 64, 32)
    tf.print('*17', tf.math.reduce_max(conv5_bn_2), tf.math.reduce_min(conv5_bn_2), tf.math.reduce_mean(conv5_bn_2),tf.math.reduce_std(conv5_bn_2))

    conc6_2 = concatenate([up5_2, conv5_bn_2], axis=-1)    # (B, 128, 128, 64, 64) + (B, 128, 128, 64, 32) -> (B, 128, 128, 64, 96)
    # (B, 128, 128, 64, 96) --> (B, 256, 256, 128, 16)  + (B, 256, 256, 128, 16) --> (B, 256, 256, 128, 32)
    up6_2 = concatenate([Conv3DTranspose(8 * factor, (2, 2, 2), strides=(2, 2, 2), padding='same')
                       (conc6_2), x2_3d_bn1_2], axis=-1)
    conv6_2 = Conv3D(8 * factor, (3, 3, 3), padding='same')(up6_2)               # (B, 256, 256, 128, 16)
    conv6_ac_2 = Activation('relu')(conv6_2)
    conv6_att_2 = attention_module_3d(conv6_ac_2, 'cbam_block')
    conv6_add_2 = keras.layers.add([conv6_ac_2, conv6_att_2])
    conv6_bn_2 = BatchNormalization()(conv6_add_2)                               # (B, 256, 256, 128, 16)
    tf.print('*17', tf.math.reduce_max(conv6_bn_2), tf.math.reduce_min(conv6_bn_2), tf.math.reduce_mean(conv6_bn_2),tf.math.reduce_std(conv6_bn_2))

    conv_2nd_3d_output = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv6_bn_2)  # (B, 256, 256, 128, 1)
    '''
    # **************************************************************************************************************** #
    """
    # The encoding path for the first branch with 2D inputs:
    x2_2d = Conv2D(filters=8 * factor, kernel_size=3, strides=1, padding='same')(inputs_A_resized_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a1_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a1_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = Conv2D(filters=8 * factor, kernel_size=3, strides=1, padding='same')(x2_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a1_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a1_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = MaxPool2D(pool_size=(2, 2))(x2_2d)

    x2_2d = Conv2D(filters=16 * factor, kernel_size=3, strides=1, padding='same')(x2_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a2_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a2_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = Conv2D(filters=16 * factor, kernel_size=3, strides=1, padding='same')(x2_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a2_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a2_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = MaxPool2D(pool_size=(2, 2))(x2_2d)

    x2_2d = Conv2D(filters=32 * factor, kernel_size=3, strides=1, padding='same')(x2_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a3_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a3_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = Conv2D(filters=32 * factor, kernel_size=3, strides=1, padding='same')(x2_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a3_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a3_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = MaxPool2D(pool_size=(2, 2))(x2_2d)

    x2_2d = Conv2D(filters=64 * factor, kernel_size=3, strides=1, padding='same')(x2_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a4_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a4_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = Conv2D(filters=64 * factor, kernel_size=3, strides=1, padding='same')(x2_2d)
    x2_2d = Activation('relu')(x2_2d)
    x2_a4_2d = attention_module_2d(x2_2d, 'cbam_block')
    x2_2d = keras.layers.add([x2_2d, x2_a4_2d])
    x2_2d = BatchNormalization()(x2_2d)
    x2_2d = MaxPool2D(pool_size=(2, 2))(x2_2d)
    x2_2d_gap = GlobalAveragePooling2D()(x2_2d)
    """

    # inputs_A_to_p3d: (B, 128, 128, 4) --> (B, 128, 128, 32)
    #'''
    x2_p3d = Conv2D(filters=8 * factor, kernel_size=3, strides=1, padding='same')(inputs_A_resized_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a1_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a1_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = Conv2D(filters=8 * factor, kernel_size=3, strides=1, padding='same')(x2_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a1_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a1_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = MaxPool2D(pool_size=(2, 2))(x2_p3d)
    #tf.print('*17', tf.math.reduce_max(conv6_bn_2), tf.math.reduce_min(conv6_bn_2), tf.math.reduce_mean(conv6_bn_2),tf.math.reduce_std(conv6_bn_2))

    x2_p3d = Conv2D(filters=16 * factor, kernel_size=3, strides=1, padding='same')(x2_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a2_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a2_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = Conv2D(filters=16 * factor, kernel_size=3, strides=1, padding='same')(x2_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a2_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a2_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = MaxPool2D(pool_size=(2, 2))(x2_p3d)
    #tf.print('*18', tf.math.reduce_max(x2_p3d), tf.math.reduce_min(x2_p3d), tf.math.reduce_mean(x2_p3d),tf.math.reduce_std(x2_p3d))

    x2_p3d = Conv2D(filters=32 * factor, kernel_size=3, strides=1, padding='same')(x2_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a3_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a3_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = Conv2D(filters=32 * factor, kernel_size=3, strides=1, padding='same')(x2_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a3_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a3_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = MaxPool2D(pool_size=(2, 2))(x2_p3d)
    #tf.print('*19', tf.math.reduce_max(x2_p3d), tf.math.reduce_min(x2_p3d), tf.math.reduce_mean(x2_p3d),tf.math.reduce_std(x2_p3d))

    x2_p3d = Conv2D(filters=64 * factor, kernel_size=3, strides=1, padding='same')(x2_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a4_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a4_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = Conv2D(filters=64 * factor, kernel_size=3, strides=1, padding='same')(x2_p3d)
    x2_p3d = Activation('relu')(x2_p3d)
    x2_a4_p3d = attention_module_2d(x2_p3d, 'cbam_block')
    x2_p3d = keras.layers.add([x2_p3d, x2_a4_p3d])
    x2_p3d = BatchNormalization()(x2_p3d)
    x2_p3d = MaxPool2D(pool_size=(2, 2))(x2_p3d)
    #tf.print('*20', tf.math.reduce_max(x2_p3d), tf.math.reduce_min(x2_p3d), tf.math.reduce_mean(x2_p3d),tf.math.reduce_std(x2_p3d))

    x2_p3d_gap = GlobalAveragePooling2D()(x2_p3d)
    #'''
    """
    x1_p3d_gap_unsqueeze = Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(x1_p3d_gap)
    x1_p3d_gap_gap = GlobalAveragePooling1D()(x1_p3d_gap_unsqueeze)
    x1_p3d_gap_gap = Lambda(lambda x: keras.backend.squeeze(x, 1))(x1_p3d_gap_gap)
    x1_p3d_gap_gap = Lambda(lambda x: keras.backend.expand_dims(x, axis=0))(x1_p3d_gap_gap)

    tf.print(x2_p3d_gap.shape, "##1")
    x2_p3d_gap_unsqueeze = Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(x2_p3d_gap)
    tf.print(x2_p3d_gap_unsqueeze.shape, "##0")
    x2_p3d_gap_gap = GlobalAveragePooling1D()(x2_p3d_gap_unsqueeze)
    tf.print(x2_p3d_gap_gap.shape, "##0.1")
    x2_p3d_gap_gap = Lambda(lambda x: keras.backend.squeeze(x, -1))(x2_p3d_gap_gap)
    tf.print(x2_p3d_gap_gap.shape, "##0.2")
    x2_p3d_gap_gap = Lambda(lambda x: keras.backend.expand_dims(x, axis=0))(x2_p3d_gap_gap)
    tf.print(x2_p3d_gap_gap.shape, "##0.3")
    """
    #x1_p3d_gap_reshape = Lambda(lambda x: keras.backend.reshape(x, shape=(-1, 63, x.shape[1])))(x1_p3d_gap)
    #x1_p3d_gap_gap = GlobalAveragePooling1D()(x1_p3d_gap_reshape)
    tf.print(x2_p3d_gap.shape, "##1")
    #x2_p3d_gap_reshape = Lambda(lambda x: keras.backend.reshape(x, shape=(-1, 29, tf.shape(x)[1])))(x2_p3d_gap)
    #x2_p3d_gap_reshape = Lambda(lambda x: keras.backend.reshape(x, shape=(-1, 39, tf.shape(x)[1])))(x2_p3d_gap)
    x2_p3d_gap_reshape = Lambda(lambda x: keras.backend.reshape(x, shape=(-1, 34, tf.shape(x)[1])))(x2_p3d_gap)
    #x2_p3d_gap_reshape = Lambda(lambda x: keras.backend.reshape(x, shape=(-1, tf.shape(x)[0]//batch_size, tf.shape(x)[1])))(x2_p3d_gap)
    x2_p3d_gap_gap = GlobalAveragePooling1D()(x2_p3d_gap_reshape)
    #x2_2d_gap_reshape = Lambda(lambda x: keras.backend.reshape(x, shape=(-1, 70, x.shape[1])))(x2_2d_gap)
    #x2_2d_gap_gap = GlobalAveragePooling1D()(x2_2d_gap_reshape)
    #tf.print('*21', tf.math.reduce_max(x1_p3d_gap_gap), tf.math.reduce_min(x1_p3d_gap_gap), tf.math.reduce_mean(x1_p3d_gap_gap),tf.math.reduce_std(x1_p3d_gap_gap))
    #tf.print('*22', tf.math.reduce_max(x2_p3d_gap_gap), tf.math.reduce_min(x2_p3d_gap_gap), tf.math.reduce_mean(x2_p3d_gap_gap),tf.math.reduce_std(x2_p3d_gap_gap))
    #tf.print(x2_3d_gap.shape, x1_p3d_gap_gap.shape, x2_p3d_gap_gap.shape, x1_3d_gap.shape, "Line-565")
    #'''
    tf.print(x1_3d_gap.shape,  x2_p3d_gap_gap.shape, "Line-565")
    x_c = concatenate([x1_3d_gap, # x1_2d_gap_gap, #
                       #x1_p3d_gap_gap,
                       #x2_3d_gap,
                       #x2_2d_gap_gap,
                       x2_p3d_gap_gap
                       ], axis=-1)
    #'''
    x_d = Dense(units=1024, activation="relu")(x_c)
    x_d = Dense(units=1024, activation="relu")(x_d)
    #tf.print('*23', tf.math.reduce_max(x_d), tf.math.reduce_min(x_d), tf.math.reduce_mean(x_d),tf.math.reduce_std(x_d))
    if num_class == 2:
        units = 1
        activation = 'sigmoid'
    else:
        units = num_class
        activation = 'softmax'

    classification_1 = Dense(units=units, activation=activation, kernel_initializer='he_normal',
                                   name='classification1_output')(x_d)
    '''
    classification_2 = Dense(units=3, activation='softmax', kernel_initializer='he_normal',
                             name='classification2_output')(x_d)
    classification_3 = Dense(units=3, activation='softmax', kernel_initializer='he_normal',
                             name='classification3_output')(x_d)
    '''
    #tf.print('*24', classification_1, classification_2, classification_3)
    #classification = [classification_1, classification_2, classification_3]
    model = Model(inputs=inputs_A_resized, outputs=classification_1)
    return model

