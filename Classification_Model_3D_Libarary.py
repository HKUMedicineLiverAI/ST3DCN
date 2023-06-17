from tensorflow.keras.layers import Conv3D, Input, MaxPool3D, BatchNormalization, Dense, GlobalAveragePooling3D, Dropout,\
    Activation, concatenate, MaxPooling3D, Flatten, AveragePooling3D, Lambda, Reshape, multiply, GlobalMaxPooling3D,\
    Conv1D, Permute
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import backend as K





def ResNet_3D_ClasModel(# input_shape=(128, 128, 64, 1), 
			width=70, height=70, depth=70,
			nb_classes=2):
    # inputs = Input(shape=input_shape)
    inputs = Input(shape=(width, height, depth, 1))
    # factor = 1
    factor = 24
    conv1 = Conv3D(32 // factor, (3, 3, 3), padding='same')(inputs)
    conv1_bn = BatchNormalization()(conv1)
    conv1_ac = Activation('relu')(conv1_bn)
    conv1_2 = Conv3D(32 // factor, (3, 3, 3), padding='same')(conv1_ac)
    conv1_2_bn = BatchNormalization()(conv1_2)
    conv1_2_ac = Activation('relu')(conv1_2_bn)
    conc1 = concatenate([inputs, conv1_2_ac], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1))(conc1)

    print(pool1.shape)
    conv2 = Conv3D(64 // factor, (3, 3, 3), padding='same')(pool1)
    conv2_bn = BatchNormalization()(conv2)
    conv2_ac = Activation('relu')(conv2_bn)
    conv2_2 = Conv3D(64 // factor, (3, 3, 3), padding='same')(conv2_ac)
    conv2_2_bn = BatchNormalization()(conv2_2)
    conv2_2_ac = Activation('relu')(conv2_2_bn)
    conc2 = concatenate([pool1, conv2_2_ac], axis=4)
    pool2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1))(conc2)
    print(pool2.shape)


    conv3 = Conv3D(128 // factor, (3, 3, 3), padding='same')(pool2)
    conv3_bn = BatchNormalization()(conv3)
    conv3_ac = Activation('relu')(conv3_bn)
    conv3_2 = Conv3D(128 // factor, (3, 3, 3), padding='same')(conv3_ac)
    conv3_2_bn = BatchNormalization()(conv3_2)
    conv3_2_ac = Activation('relu')(conv3_2_bn)
    conc3 = concatenate([pool2, conv3_2_ac], axis=4)
    pool3 = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(conc3)

    conv4 = Conv3D(256 // factor, (3, 3, 3), padding='same')(pool3)
    conv4_bn = BatchNormalization()(conv4)
    conv4_ac = Activation('relu')(conv4_bn)
    conv4_2 = Conv3D(256 // factor, (3, 3, 3), padding='same')(conv4_ac)
    conv4_2_bn = BatchNormalization()(conv4_2)
    conv4_2_ac = Activation('relu')(conv4_2_bn)
    conc4 = concatenate([pool3, conv4_2_ac], axis=4)
    pool4 = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3))(conc4)

    conv5 = Conv3D(512 // factor, (3, 3, 3), padding='same')(pool4)
    conv5_bn = BatchNormalization()(conv5)
    conv5_ac = Activation('relu')(conv5_bn)
    conv5_2 = Conv3D(512 // factor, (3, 3, 3), padding='same')(conv5_ac)
    conv5_2_bn = BatchNormalization()(conv5_2)
    conv5_2_ac = Activation('relu')(conv5_2_bn)
    conc5 = concatenate([pool4, conv5_2_ac], axis=4)

    # x = BatchNormalization()(conc4)
    # x = Activation('relu')(x)
    x = conc5
    x = AveragePooling3D(pool_size=(4, 4, 4), strides=(4, 4, 4))(x)

    # y = GlobalAveragePooling3D()4

    y = Flatten()(x)
    # y = Dense(units=512, activation='softmax', kernel_initializer='he_normal')(y)
    # y = Dense(units=512, activation='softmax', kernel_initializer='he_normal')(y)
    if nb_classes == 2:
        units = 1
        activation = 'sigmoid'
    else:
        units = nb_classes
        activation = 'softmax'

    outputs = Dense(units=units, activation=activation, kernel_initializer='he_normal')(y)
    # outputs = Dense(units=nb_classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# model = ResNet_3D_ClasModel()
# print(model.summary())




def attention_module(net, attention_module):
    if attention_module == 'se_block':  # SE_Block
        net = se_block(net)
    elif attention_module == 'cbam_block':
        net = cbam_block(net)
    elif attention_module == 'eca_block':
        net = eca_block(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # channel = input_feature._keras_shape[channel_axis]
    channel = input_feature.shape[channel_axis]
    print(channel_axis, channel)
    se_feature = GlobalAveragePooling3D()(input_feature)
    se_feature = Reshape((1, 1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, 1, channel)

    se_feature = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, 1, channel//ratio)

    se_feature = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, 1, channel)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


def cbam_block(cbam_feature, ratio=8):
    # cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = eca_block(cbam_feature)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel)

    max_pool = GlobalMaxPooling3D()(input_feature)
    max_pool = Reshape((1, 1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel)

    cbam_feature = keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 3
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    avg_pool = Lambda(lambda x: K.mean(x, axis=channel_axis, keepdims=True))(input_feature)
    max_gool = Lambda(lambda x: K.max(x, axis=channel_axis, keepdims=True))(input_feature)
    concat = keras.layers.Concatenate(axis=channel_axis)([avg_pool, max_gool])
    cbam_feature = Conv3D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])


def eca_block(input_feature):
    avg_pool = GlobalAveragePooling3D()(input_feature)    # [B, H, W, D, Channels] --> [B, channels]
    unsqueezed_avg_pool = Lambda(lambda x: keras.backend.expand_dims(x, 1))(avg_pool)   # ---> [B, 1, channels]
    permuted_avg_pool = Permute((2, 1))(unsqueezed_avg_pool)     # ---> [B, channels, 1]
    # squeezed_avg_pool_w = Lambda(lambda x: keras.backend.squeeze(x, 1))(squeezed_avg_pool_h)  # ---> [B, 1, channels]
    # The output of Conv1D is [B, channels, 1]
    conv_avg_pool = Conv1D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(permuted_avg_pool)
    permuted_conv_output = Permute((2, 1))(conv_avg_pool)  # ---> [B, channels, 1]
    unsqueezed_w = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(permuted_conv_output)   # ---> [B, 1, channels, 1]
    print(unsqueezed_w)
    unsqueezed_h = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(unsqueezed_w)    # ---> [B, 1, 1, channels, 1]
    print(unsqueezed_h)

    output_feature = multiply([input_feature, unsqueezed_h])
    return output_feature


def get_model_CABM(width=256, height=256, depth=128, num_class=3):
    inputs = Input((width, height, depth, 1))
    factor = 2
    x = Conv3D(filters=8*factor, kernel_size=3, strides=1)(inputs)
    x = Activation('relu')(x)
    x_a1 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a1])
    x = BatchNormalization()(x)
    x = Conv3D(filters=8*factor, kernel_size=3, strides=1)(x)
    x_a1 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a1])
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)    # change from 3 to 2

    x = Conv3D(filters=16*factor, kernel_size=3, strides=1)(x)
    x = Activation('relu')(x)
    x_a2 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a2])
    x = BatchNormalization()(x)
    x = Conv3D(filters=16*factor, kernel_size=3, strides=1)(x)
    x_a2 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a2])
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(3, 3, 3))(x)

    x = Conv3D(filters=32*factor, kernel_size=3, strides=1)(x)
    x = Activation('relu')(x)
    x_a3 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a3])
    x = BatchNormalization()(x)
    x = Conv3D(filters=32*factor, kernel_size=3, strides=1)(x)
    x_a3 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a3])
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=(3, 3, 3))(x)

    x = Conv3D(filters=64*factor, kernel_size=3, strides=1)(x)
    x = Activation('relu')(x)
    x_a3 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a3])
    x = BatchNormalization()(x)
    x = Conv3D(filters=64*factor, kernel_size=3, strides=1)(x)
    x_a3 = attention_module(x, 'cbam_block')
    x = keras.layers.add([x, x_a3])
    x = BatchNormalization()(x)
    # x = MaxPool3D(pool_size=(3, 3, 3))(x)

    x = GlobalAveragePooling3D()(x)
    # x = Flatten()(x)
    x = Dense(units=512, activation="relu")(x)
    # x = Dropout(0.5)(x)
    x = Dense(units=512, activation="relu")(x)

    if num_class == 2:
        outputs = Dense(units=1, activation="sigmoid")(x)
    else:
        outputs = Dense(units=num_class, activation="softmax")(x)

    model = Model(inputs, outputs, name="3dcnn")
    return model


def FCN_3DSE(# input_shape=(128, 128, 64, 1), nb_classes=2
           width = 70, height = 70, depth = 70,
           nb_classes = 2):
    # inputs = Input(shape=input_shape, name='inputs')
    inputs = Input(shape=(width, height, depth, 1))
    # Block 1:
    factor = 2
    block1_conv1 = Conv3D(64//factor, (3, 3, 3), padding='same', name='block1_conv1')(inputs)
    block1_att1 = attention_module(block1_conv1, 'se_block')
    block1_add1 = keras.layers.add([block1_conv1, block1_att1])
    block1_bn1 = BatchNormalization()(block1_add1)
    block1_ac1 = Activation('relu')(block1_bn1)
    block1_conv2 = Conv3D(64//factor, (3, 3, 3), padding='same', name='block1_conv2')(block1_ac1)
    block1_att1 = attention_module(block1_conv2, 'se_block')
    block1_add2 = keras.layers.add([block1_conv2, block1_att1])
    block1_bn2 = BatchNormalization()(block1_add2)
    block1_ac2 = Activation('relu')(block1_bn2)
    block1_pool = MaxPooling3D((2, 2, 1), strides=(1, 1, 1), name='block1_pool')(block1_ac2)
    print(block1_pool.shape)
    # Block 2:
    block2_conv1 = Conv3D(128//factor, (3, 3, 3), padding='same', name='block2_conv1')(block1_pool)
    block2_att1 = attention_module(block2_conv1, 'se_block')
    block2_add1 = keras.layers.add([block2_conv1, block2_att1])
    block2_bn1 = BatchNormalization()(block2_add1)
    block2_ac1 = Activation('relu')(block2_bn1)
    block2_conv2 = Conv3D(128//factor, (3, 3, 3), padding='same', name='block2_conv2')(block2_ac1)
    block2_att2 = attention_module(block2_conv2, 'se_block')
    block2_add2 = keras.layers.add([block2_conv2, block2_att2])
    block2_bn2 = BatchNormalization()(block2_add2)
    block2_ac2 = Activation('relu')(block2_bn2)
    block2_pool = MaxPooling3D((2, 2, 2), name='block2_pool')(block2_ac2)

    # Block 3:
    block3_conv1 = Conv3D(256//factor, (3, 3, 3), padding='same', name='block3_conv1')(block2_pool)
    block3_att1 = attention_module(block3_conv1, 'se_block')
    block3_add1 = keras.layers.add([block3_conv1, block3_att1])
    block3_bn1 = BatchNormalization()(block3_add1)
    block3_ac1 = Activation('relu')(block3_bn1)
    block3_conv2 = Conv3D(256//factor, (3, 3, 3), padding='same', name='block3_conv2')(block3_ac1)
    block3_att2 = attention_module(block3_conv2, 'se_block')
    block3_add2 = keras.layers.add([block3_conv2, block3_att2])
    block3_bn2 = BatchNormalization()(block3_add2)
    block3_ac2 = Activation('relu')(block3_bn2)
    block3_pool = MaxPooling3D((2, 2, 2), name='block3_pool')(block3_ac2)
    print(block3_pool.shape)
    # Block 4:
    block4_conv1 = Conv3D(512//factor, (3, 3, 3), padding='same', name='block4_conv1')(block3_pool)
    block4_att1 = attention_module(block4_conv1, 'se_block')
    block4_add1 = keras.layers.add([block4_conv1, block4_att1])
    block4_bn1 = BatchNormalization()(block4_add1)
    block4_ac1 = Activation('relu')(block4_bn1)
    block4_conv2 = Conv3D(512//factor, (3, 3, 3), padding='same', name='block4_conv2')(block4_ac1)
    block4_att2 = attention_module(block4_conv2, 'se_block')
    block4_add2 = keras.layers.add([block4_conv2, block4_att2])
    block4_bn2 = BatchNormalization()(block4_add2)
    block4_ac2 = Activation('relu')(block4_bn2)
    block4_pool = MaxPooling3D((2, 2, 2), name='block4_pool')(block4_ac2)
    print(block4_pool.shape)
    # Block 5:
    # y = GlobalAveragePooling3D()(block4_pool)   #
    y = Flatten()(block4_pool)
    print(y.shape)
    # y = GlobalAveragePooling3D()(block4_conv3_bn)
    y = Dense(512, activation='relu', kernel_initializer='he_normal')(y)
    # y = Dropout(0.5)(y)
    y = Dense(512, activation='relu', kernel_initializer='he_normal')(y)
    if nb_classes == 2:
        units = 1
        activation = 'sigmoid'
    else:
        units = nb_classes
        activation = 'softmax'

    outputs = Dense(units=units, activation=activation, kernel_initializer='he_normal')(y)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model



def C3D_Net_Model(# input_shape=(128, 128, 64, 1), nb_classes=2
    width = 70, height = 70, depth = 70,
    nb_classes = 2
):
    # inputs = Input(shape=input_shape, name='inputs')
    inputs = Input(shape=(width, height, depth, 1))
    factor = 1

    block1_conv1 = Conv3D(32*factor, (3, 3, 3), padding='same', name='block1_conv1')(inputs)
    block1_bn1 = BatchNormalization()(block1_conv1)
    block1_ac1 = Activation('relu')(block1_bn1)
    block1_pool1 = AveragePooling3D((2, 2, 1), strides=(2, 2, 1))(block1_ac1)
    block1_conv2 = Conv3D(64*factor, (3, 3, 3), padding='same', name='block1_conv2')(block1_pool1)
    block1_bn2 = BatchNormalization()(block1_conv2)
    block1_ac2 = Activation('relu')(block1_bn2)
    block1_pool2 = AveragePooling3D((2, 2, 2), strides=(1, 1, 1))(block1_ac2)

    block2_conv1 = Conv3D(128*factor, (3, 3, 3), padding='same', name='block2_conv1')(block1_pool2)
    block2_bn1 = BatchNormalization()(block2_conv1)
    block2_ac1 = Activation('relu')(block2_bn1)
    block2_pool1 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(block2_ac1)
    block2_conv2 = Conv3D(128*factor, (3, 3, 3), padding='same', name='block2_conv2')(block2_pool1)
    block2_bn2 = BatchNormalization()(block2_conv2)
    block2_ac2 = Activation('relu')(block2_bn2)
    block2_pool2 = AveragePooling3D((2, 2, 2), strides=(1, 1, 1))(block2_ac2)

    block3_conv1 = Conv3D(256*factor, (3, 3, 3), padding='same', name='block3_conv1')(block2_pool2)
    block3_bn1 = BatchNormalization()(block3_conv1)
    block3_ac1 = Activation('relu')(block3_bn1)
    block3_pool1 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(block3_ac1)
    block3_conv2 = Conv3D(256*factor, (3, 3, 3), padding='same', name='block3_conv2')(block3_pool1)
    block3_bn2 = BatchNormalization()(block3_conv2)
    block3_ac2 = Activation('relu')(block3_bn2)
    block3_pool2 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(block3_ac2)

    block4_conv1 = Conv3D(256*factor, (3, 3, 3), padding='same', name='block4_conv1')(block3_pool2)
    block4_bn1 = BatchNormalization()(block4_conv1)
    block4_ac1 = Activation('relu')(block4_bn1)
    block4_pool1 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(block4_ac1)
    block4_conv2 = Conv3D(256*factor, (3, 3, 3), padding='same', name='block4_conv2')(block4_pool1)
    block4_bn2 = BatchNormalization()(block4_conv2)
    block4_ac2 = Activation('relu')(block4_bn2)
    block4_pool2 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(block4_ac2)

    y = Flatten()(block4_pool2)

    y = Dense(512, activation='relu', kernel_initializer='he_normal')(y)
    # y = Dropout(0.5)(y)
    y = Dense(512, activation='relu', kernel_initializer='he_normal')(y)
    if nb_classes == 2:
        units = 1
        activation = 'sigmoid'
    else:
        units = nb_classes
        activation = 'softmax'

    outputs = Dense(units=units, activation=activation, kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model





