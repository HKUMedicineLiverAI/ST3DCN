import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#from keras.utils import multi_gpu_model
#from CT_Scans_Data_Augmentation import train_preprocessing, validation_preprocessing
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras import backend as K
tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
#config = tf.compat.v1.ConfigProto
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True
#sess = tf.compat.v1.Session(config=config)
#sess = tf.Session(config=config)
#K.set_session(sess)
import keras
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from keras.optimizers import Adam, SGD
from MS3DCN_Utils_patient_31 import multi_scale_get_model_DCN
import pickle
from operator import itemgetter
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
Window_Level = (400, 40)
import nibabel as nib
import random
import scipy
import datetime


smooth = 1.0e-6


#tf.config.experimental_run_functions_eagerly(True)
def data_augmentation(image3d):
    image_3d = image3d
    flip_num = np.random.randint(0, 20)
    #print('#0', flip_num)
    if flip_num == 1:
        #image_3d = np.flipud(image_3d)  # Flip array in the up/down direction.symmetric
        image_3d = np.flip(image_3d, axis=0)
    elif flip_num == 2:
        #image_3d = np.fliplr(image_3d)  # # Flip array in the left/right direction.
        image_3d = np.flip(image_3d, axis=1)
    elif flip_num == 3:
        image_3d = np.rot90(image_3d, k=1, axes=(0, 1))  # 256, 256, 3
    elif flip_num == 4:
        image_3d = np.rot90(image_3d, k=3, axes=(0, 1))

    elif flip_num == 5:
        #image_3d = np.fliplr(image_3d)
        image_3d = np.flip(image_3d, axis=0)
        image_3d = np.rot90(image_3d, k=1, axes=(0, 1))
    elif flip_num == 6:
        #image_3d = np.fliplr(image_3d)
        image_3d = np.flip(image_3d, axis=0)
        image_3d = np.rot90(image_3d, k=3, axes=(0, 1))

    elif flip_num == 7:
        #image_3d = np.flipud(image_3d)
        #image_3d = np.fliplr(image_3d)
        image_3d = np.flip(image_3d, axis=0)
        image_3d = np.flip(image_3d, axis=1)
    elif flip_num == 8:
        image_3d = random_rotation_3d(image_3d, 90, 1)
    elif flip_num == 9:
        image_3d = random_rotation_3d(image_3d, 90, 2)
    elif flip_num == 10:
        image_3d = random_rotation_3d(image_3d, 90, 3)
    elif flip_num == 11:
        image_3d = random_rotation_3d(image_3d, 90, 4)
    elif flip_num == 12:
        image_3d = random_rotation_3d(image_3d, 90, 5)
    elif flip_num == 13:
        image_3d = random_rotation_3d(image_3d, 90, 6)
    elif flip_num == 14:
        image_3d = random_rotation_3d(image_3d, 90, 7)
    else: pass

    return image_3d

def random_rotation_3d(batch_image, max_angle, flag_num):
    """
    :param batch:
    :param max_angle:  the maximum rotation angle
    :return:
    """
    size = batch_image.shape           # (H, W, Depth)
    #batch_image_squeeze = np.squeeze(batch_image, axis=-1)    # (B, H, W, Depth)
    #batch_mask_squeeze = np.squeeze(batch_mask, axis=-1)
    batch_image_rot = np.zeros(shape=size)     # (B, H, W, Depth)
    #batch_mask_rot = np.zeros(shape=batch_mask_squeeze.shape)
    #print('##', batch_image_squeeze.shape[0])
    #for i in range(batch_image_squeeze.shape[0]):
        #if bool(random.getrandbits(1)):
        #print("#1", 1)
    #image1_0 = np.squeeze(batch_image_squeeze[i])       #  batch[i]: (1, 256, 256, 128) --> (256, 256, 128)
    #image1_1 = np.squeeze(batch_mask_squeeze[i])
    #flag_num = np.random.randint(1, 8)
    angle = random.uniform(-max_angle, max_angle)
    if flag_num==1:
        # rotate along z-axis
            #angle = random.uniform(-max_angle, max_angle)
        batch_image_rot = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 1), reshape=False)
    elif flag_num==2:
            # rotate along y-axis
            #angle = random.uniform(-max_angle, max_angle)
        batch_image_rot = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 2), reshape=False)
        #batch_mask_rot = scipy.ndimage.rotate(image1_1, angle, mode='nearest', axes=(0, 2), reshape=False)
    elif flag_num==3:
            # rotate along x-axis
            #angle = random.uniform(-max_angle, max_angle)
        batch_image_rot = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(1, 2), reshape=False)
        #batch_mask_rot = scipy.ndimage.rotate(image1_1, angle, mode='nearest', axes=(1, 2), reshape=False)
    elif flag_num==4:
        batch_image = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 1), reshape=False)
        angle = random.uniform(-max_angle, max_angle)
        batch_image_rot = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 2), reshape=False)
    elif flag_num==5:
        batch_image = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 1), reshape=False)
        angle = random.uniform(-max_angle, max_angle)
        atch_image_rot = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(1, 2), reshape=False)
    elif flag_num==6:
        batch_image = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 1), reshape=False)
        angle = random.uniform(-max_angle, max_angle)
        batch_image_rot = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(1, 2), reshape=False)
    elif flag_num==7:
        batch_image = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 1), reshape=False)
        angle = random.uniform(-max_angle, max_angle)
        batch_image = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(0, 2), reshape=False)
        angle = random.uniform(-max_angle, max_angle)
        batch_image_rot = scipy.ndimage.rotate(batch_image, angle, mode='nearest', axes=(1, 2), reshape=False)
    else: pass
            #print("#1", 0)
            #batch_image_rot[i] = batch_image_squeeze[i]
            #batch_mask_rot[i] = batch_mask_squeeze[i]

    #reshaped_batch_image, reshape_batch_mask = batch_image_rot.reshape(size), batch_mask_rot.reshape(size)
    return batch_image_rot

def padzero(in_img, new_x, new_y,new_z):
    old_x, old_y,old_z=in_img.shape

    x_center, y_center, z_center=0,0,0
    if old_x < new_x:
        x_center = (new_x - old_x) // 2
    else:
        new_x = old_x

    if old_y < new_y:
        y_center = (new_y - old_y) // 2
    else:
        new_y = old_y

    if old_z < new_z:
        z_center = (new_z - old_z) // 2
    else:
        new_z = old_z

    out_img = np.zeros((new_x, new_y,new_z))
    out_img[x_center:x_center+old_x,
           y_center:y_center+old_y,
           z_center:z_center+old_z] = in_img

    return out_img

def resize_volume(img, desired_height, desired_width, desired_depth):
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[1]
    current_height = img.shape[0]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    #img = np.flipud(img)  # add np.flipup here 20221223:11:11AM
    # Resize across z-axis
    img = ndimage.zoom(img, (height_factor, width_factor, depth_factor), order=1)
    #img = ndimage.zoom(img, (height_factor, width_factor, depth_factor))
    return img

class CT3D_DataLoader_For_Seg_Clas(keras.utils.Sequence):
    def __init__(self, batch_size=4,
                 image_depth=64,
                 image_size=(128, 128),
                 input_data_path=None,
                 num_class=2,
                 training_or_testing='',
                 shuffle=True):
        self.batch_size = batch_size
        self.image_depth = image_depth
        self.image_size = image_size
        self.input_data_path = input_data_path
        self.num_class = num_class
        self.training_or_testing = training_or_testing
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        return len(self.input_data_path)//self.batch_size

    def on_epoch_end(self):
        'Updates indexs after each epoch'
        self.indexes=np.arange(len(self.input_data_path))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)    
        
        #print('%1',self.training_or_testing, self.indexes)

    def __getitem__(self, idx):
        '''
        self.indexes = np.arange(len(self.input_data_path))

        if self.training_or_testing == 'training':
            np.random.shuffle(self.indexes)
            self.input_data_path_shuffled = self.input_data_path[self.indexes]
        else:
            self.input_data_path_shuffled = self.input_data_path
        '''
        #i = idx * self.batch_size
        indexes = self.indexes[idx * self.batch_size:(idx + 1)* self.batch_size]
        batch_data_path = [self.input_data_path[k] for k in indexes]
        #print('#0', self.training_or_testing, idx, len(indexes), indexes, len(batch_data_path), batch_data_path)
        #batch_data_path = self.input_data_path_shuffled[i:i+self.batch_size]

        data_batch = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_depth, 1), dtype="float32")
        label_batch = np.zeros((self.batch_size, ), dtype='float32')

        for i in range(len(batch_data_path)):
            #vol_scans = np.load(batch_data_path[i][0]) / 255.0
            #print('#1', self.training_or_testing,idx, len(batch_data_path), batch_data_path[i][0], batch_data_path[i][1])
            vol_scans = np.load(batch_data_path[i][1])

            max_size = max(vol_scans.shape)
            min_size = min(vol_scans.shape)
            if min_size < self.image_size[0]:
                vol_scans = padzero(vol_scans, self.image_size[0], self.image_size[1],self.image_depth)
            if max_size > self.image_size[0]:
                vol_scans = resize_volume(vol_scans, self.image_size[0], self.image_size[1],self.image_depth)
            vol_scans = vol_scans / 255.0
            if self.training_or_testing == 'training':
                vol_scans_aug = data_augmentation(vol_scans)
                vol_scans_aug[vol_scans_aug>1]=1
                vol_scans_aug[vol_scans_aug<0]=0
            else:
                vol_scans_aug = vol_scans
            data_batch[i] = np.expand_dims(vol_scans_aug, axis=-1)
            label_batch[i] = int(batch_data_path[i][-1])
            #print('#r1',batch_data_path[i][0], first_lesion_label_batch[i], second_lesion_label_batch[i], third_lesion_label_batch[i])

        # data_batch = np.expand_dims(data_batch, axis=-1)


        #first_lesion_label_batch = first_lesion_label_batch
        #y2 = keras.utils.to_categorical(second_lesion_label_batch, num_classes=3)
        #y3 = keras.utils.to_categorical(third_lesion_label_batch, num_classes=3)
        #data_batch_nor = (data_batch - np.amin(data_batch, axis=0)) / (np.amax(data_batch, axis=0) - np.amin(data_batch, axis=0))
        #for i in range(data_batch.shape[0]):
        #    print('#r2', np.amax(data_batch[i]), np.amin(data_batch[i]))
        #print('#e3', np.amin(data_batch, axis=0), np.amax(data_batch, axis=0))
        return data_batch, label_batch

#Train_Path_Lesion_Label_List = np.load(os.path.join('/home/ra1/original', 'Train_Path_Lesion_Label_List.npy'))
#Test_Path_Lesion_Label_List = np.load(os.path.join('/home/ra1/original', 'Test_Path_Lesion_Label_List.npy'))
#Train_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/patient_level_augmentation/P3', 'P3_train_aug_nor.npy'), allow_pickle=True)
#Test_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/patient_level_augmentation/P3', 'P3_org_validation_normalized.npy'), allow_pickle=True)
#Train_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/patient_level_augmentation/', 'P2_2468_P3_13579_train_aug_nor.npy'), allow_pickle=True)
#Test_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/patient_level_augmentation/', 'P2_P3_org_validation_normalized.npy'), allow_pickle=True)
#Train_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/resize/training', 'P2_P3_registrated_nor_resized.npy'), allow_pickle=True)
#Test_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/resize/testing', 'P2_P3_registed_nor_resized.npy'), allow_pickle=True)

#Train_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/resize/training', 'P2_P3_registrated_nor_resized_remove_mislabed.npy'), allow_pickle=True)
#Test_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/patient_level/resize/testing', 'P2_P3_registed_nor_resized_remove_mislabed.npy'), allow_pickle=True)
#Train_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/dataset_20230305', 'P3_P2_20230305_All_Mask_Data_Fullpath_List_prepro_train.npy'), allow_pickle=True)
#Test_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/dataset_20230305', 'P3_P2_20230305_All_Mask_Data_Fullpath_List_prepro_test.npy'), allow_pickle=True)
Train_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/crop_20230401', 'P2_P3_20230401_All_Mask_Data_Fullpath_List_crop_train.npy'), allow_pickle=True)
Test_Path_Lesion_Label_List = np.load(os.path.join('/home/GPU/jllu/work/project/hccai/data/crop_20230401', 'P2_P3_20230401_All_Mask_Data_Fullpath_List_crop_test.npy'), allow_pickle=True)

train_sample_num = Train_Path_Lesion_Label_List.shape[0]
print('#0', train_sample_num, Train_Path_Lesion_Label_List.shape)
#train = train.astype(float)
(class1_0, class1_1) = tuple(np.bincount(Train_Path_Lesion_Label_List[:,-1].astype(int).flatten()))

print('#train 1', class1_0, class1_1, Train_Path_Lesion_Label_List.shape)

test_sample_num = Test_Path_Lesion_Label_List.shape[0]
print('#0', test_sample_num, Test_Path_Lesion_Label_List.shape)
#train = train.astype(float)
(test_class1_0, test_class1_1) = tuple(np.bincount(Test_Path_Lesion_Label_List[:,-1].astype(int).flatten()))

print('#test 1', test_class1_0, test_class1_1, Test_Path_Lesion_Label_List.shape)


class1_0_ratio, class1_1_ratio=(1 / class1_0) * (train_sample_num / 2.0), \
                               (1 / class1_1) * (train_sample_num / 2.0)

print('#train 4', class1_0_ratio, class1_1_ratio)

'''
class_weight = {'classification1_output': {0:class1_0_ratio, 1:class1_1_ratio},
                'classification2_output': {0:class2_0_ratio, 1:class2_1_ratio, 2:class2_2_ratio},
                'classification3_output': {0:class3_0_ratio, 1:class3_1_ratio, 2:class3_2_ratio}}
'''
batch_size = 16
image_depth, image_rows, image_cols = 70, 70, 70
train_gen = CT3D_DataLoader_For_Seg_Clas(batch_size=batch_size,
                                         image_depth=image_depth,
                                         image_size=(image_rows, image_cols),
                                         input_data_path=Train_Path_Lesion_Label_List,
                                         num_class=2, training_or_testing='training', shuffle=True)
val_gen = CT3D_DataLoader_For_Seg_Clas(batch_size=batch_size,
                                       image_depth=image_depth,
                                       image_size=(image_rows, image_cols),
                                       input_data_path=Test_Path_Lesion_Label_List,
                                       num_class=2,
                                       training_or_testing='validation or testing',shuffle=False)

number_GPU = 1
model = multi_scale_get_model_DCN(width=image_rows, height=image_cols, depth=image_depth, factor=8, num_gpu=number_GPU, batch_size=batch_size, num_class=2)
model.summary()
#keras.utils.plot_model(model, show_shapes=True)
# Compile model.

data_rootpath = '/home/GPU/jllu/work/project/hccai/ms3dcn/patient'
if not os.path.exists(os.path.join(data_rootpath, 'patient_Model_31')):
    os.makedirs(os.path.join(data_rootpath, 'patient_Model_31'))

# Compile model.
"""
losses = {'segmentation_output': dice_coef_loss,
          'classification_output': total_classification,
          }
lossWeights = {'segmentation_output': 1.0,
               'classification_output': 0.8}
model.compile(optimizer=SGD(lr=0.01), loss=losses, loss_weights=lossWeights, metrics=['Accuracy'])
"""
metrics_dic = {'classification1_output': keras.metrics.binary_accuracy,
               'classification2_output': keras.metrics.categorical_accuracy,
               'classification3_output': keras.metrics.categorical_accuracy}

initial_learning_rate = 0.0001
'''
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
'''
#optimizer = SGD(learning_rate=initial_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
global_step = tf.Variable(0, trainable=False)
lr_scheduar = tf.compat.v1.train.exponential_decay(learning_rate=initial_learning_rate,global_step=global_step, decay_steps= 100000, decay_rate=0.96)
optimizer = Adam(learning_rate=lr_scheduar)
model.compile(#optimize.=SGD(lr=0.01),
              optimizer=optimizer,
              loss=keras.metrics.binary_crossentropy,
              metrics=[keras.metrics.binary_accuracy,
              tf.keras.metrics.AUC(),
              tf.keras.metrics.Precision(),
              tf.keras.metrics.Recall()]
              )

# Define callbacks.
checkpoint_cb = ModelCheckpoint(os.path.join(data_rootpath,
                                             'patient_Model_31/MS3DCN_Classification_Model_patient_31_70_auc4.h5'),
                                monitor='val_auc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
early_stopping_cb = EarlyStopping(monitor="val_loss", min_delta=0, patience=40, verbose=1, mode='auto')

if not os.path.exists(os.path.join(data_rootpath, 'patient_Model_31/logs')):
    os.makedirs(os.path.join(data_rootpath, 'patient_Model_31/logs'))

log_dir = "patient_Model_31/logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#callbacks_list = [checkpoint_cb, early_stopping_cb]
callbacks_list = [checkpoint_cb, early_stopping_cb,tensorboard_callback]

#keras.backend.set_session(
#    tf_debug.LocalCLIDebugWrapperSession(tf.Session())) #"cais0:7000"

epochs = 120
class_weight= {0:class1_0_ratio, 1:class1_1_ratio}
model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=2, workers=4,class_weight=class_weight,
          callbacks=callbacks_list)
if not os.path.exists(os.path.join(data_rootpath, 'patient_Model_31')):
    os.makedirs(os.path.join(data_rootpath, 'patient_Model_31'))
model.save_weights(os.path.join(data_rootpath, 'patient_Model_31/Finalized_MS3DCN_Model_HCC_patient_31_70_auc4.h5'))


