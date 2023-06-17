import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from nibabel.processing import resample_to_output, resample_from_to
from scipy.ndimage import zoom
from skimage.morphology import remove_small_holes, binary_dilation, binary_erosion, ball
from skimage.measure import label, regionprops
import copy
import os
import pandas as pd
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import nibabel as nib
import SimpleITK as sitk
import cv2
from nibabel.processing import resample_to_output
from skimage.morphology import remove_small_holes
import random
import scipy
import sys
from scipy import ndimage
import copy

Window_Level = (400, 40)
ranseed = 1234
def read_nifti(file_dir):
    file = nib.load(file_dir)
    file_voxels = file.get_data()
    #file_voxels = np.transpose(file_voxels, (2, 0, 1))
    file_hdr = file.header
    file_affine = file._affine
    return file_voxels, file_hdr, file_affine


def read_mask(file_dir):
    seg_img = sitk.ReadImage(file_dir)
    vox_data = sitk.GetArrayFromImage(seg_img)
    vox_dir = seg_img.GetDirection()
    if vox_dir[0] < 0:
        vox_data = np.flip(vox_data, axis=0)
    if vox_dir[4] < 0:
        vox_data = np.flip(vox_data, axis=1)

    return vox_data


def read_dicom(file_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(file_dir)
    reader.SetFileNames(dicom_names)
    dicom_img = reader.Execute()
    vol_data = sitk.GetArrayFromImage(img_trans(sitk.Cast(dicom_img, sitk.sitkFloat32)))
    vol_dir = dicom_img.GetDirection()

    return vol_data, vol_dir


def img_trans(img):
    new_img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=Window_Level[1] - Window_Level[0]/2.0,
                                                windowMaximum=Window_Level[1] + Window_Level[0]/2),
                        sitk.sitkUInt8)
    return new_img

def intensity_normalization(volume, intensity_clipping_range):
    result = np.copy(volume)

    result[volume < intensity_clipping_range[0]] = intensity_clipping_range[0]
    result[volume > intensity_clipping_range[1]] = intensity_clipping_range[1]

    min_val = np.amin(result)
    max_val = np.amax(result)
    if (max_val - min_val) != 0:
        result = (result - min_val) / (max_val - min_val)

    return result

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

def resize_volume(img, depth=128, width=128, height=128):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = depth
    desired_width = width
    desired_height = height
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
def draw(files, idx=0,top_slice=0,bot_slice=0):
    img_path = files[idx][1]
    mask_path =  files[idx][0]
    img_path = '/home/GPU/HCC_Orig_CT_All/PYN_Part2/Phase3_data/ID_0848_P3'
    mask_path = '/home/GPU/HCC_Orig_Mask_All/2023-02-20_updated_Mask/ID_0848_P3.nii.gz'
    print('path:', img_path, mask_path, sep='\n')
     # import image volume
    img, _ = read_dicom(img_path)
    if np.char.find(np.array(mask_path), 'gz') != -1:
        mask = read_mask(mask_path)  # the shape of output is [512, 512, num_slices]
        mask = mask.transpose(2, 1, 0)
    else:
        mask = read_nifti(mask_path)[0]
    #print('#2',img.shape, mask.shape)
    vol_data = img.transpose(2, 1, 0)  # Data and Mask with the shape of (H, W, Depth)
    vol_data = vol_data.transpose(1, 0, 2)
    vol_data = vol_data/ 255.0
    #mask_data = mask.transpose(2, 1, 0)
    mask_data = mask.transpose(1, 0, 2)
    mask_data[mask_data>0]=1
    mask_data = (mask_data > 0).astype('float32')
    print('shape of image and mask',vol_data.shape, mask_data.shape)

    for slice in range(top_slice,bot_slice):
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(vol_data[:,:,slice], cmap='gray')
        ax[0].axis('off')

        ax[1].imshow(vol_data[:,:,slice], cmap='gray')
        ax2_array=copy.deepcopy(mask_data[:,:,slice])
        ax2_array[ax2_array>0]=1
        axbar2 = ax[1].imshow(mask_data[:,:,slice], cmap='jet', alpha=ax2_array, vmin=0,vmax=1)
        ax[1].axis('off')
        fig.tight_layout()
    print('end')    #img = ndimage.rotate(img, 90, reshape=False)
    #img = np.flipud(img)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def predict_lesion(model, patient_label, lesion_data):
    batch_size = 16
    if len(lesion_data) == batch_size*int(len(lesion_data)/batch_size):
        num_batches = int(len(lesion_data)/batch_size)
    else:
        num_batches = int(len(lesion_data)/batch_size) + 1
    all_patient_id_list = []
    y_pred_ms3d, y_label_ms3d = [], []
    pred_P2_P3, all_label_P2_P3 = {},{}
    prob_accord = {}
    for i in range(num_batches):
        if i*batch_size+batch_size <= len(lesion_data):
            batch = lesion_data[i*batch_size: i*batch_size+batch_size]
        else:
            batch = lesion_data[i*batch_size: len(lesion_data)]
    x_data_batch = []
    patient_id_list = []
    #print('#111', batch.shape)
    for j in range(batch.shape[0]):
        tmp = np.load(batch[j][1])
        max_size = max(tmp.shape)
        min_size = min(tmp.shape)
        if min_size < width:
            tmp = padzero(tmp, width, height,depth)
        if max_size > width:
            tmp = resize_volume(tmp, width, height,depth)
        tmp = tmp / 255.0
        tmp[tmp>1]=1
        tmp[tmp<0]=0

        tmp = np.expand_dims(tmp, axis=0)
        x_data_batch.append(tmp)
        sample_id = os.path.basename(batch[j][0]).split('.np')[0]
        sample_id1 = sample_id.split('_P')[0] + sample_id.split('_P')[1][1:]
        patient_id_list.append(sample_id1)
        all_patient_id_list.append(sample_id)
        if sample_id1 in all_label_P2_P3:
            assert all_label_P2_P3[sample_id1] == int(batch[j][-1])
        else:
            all_label_P2_P3[sample_id1] = int(batch[j][-1])
        y_label_ms3d.append(int(batch[j][-1]))
        #Testing_ID_GTLabel.append([sample_id, int(batch[j][-1])])

    x_data_batch = np.array(x_data_batch).reshape((len(x_data_batch), width, height, depth))
    x_data_batch = np.expand_dims(x_data_batch, axis=-1)

    y_pred_temp2_ms3d = model.predict(x_data_batch).ravel()
    for j in range(batch.shape[0]):
        y_pred_ms3d.append(y_pred_temp2_ms3d[j])
        if patient_id_list[j] in pred_P2_P3:
            if y_pred_temp2_ms3d[j] > pred_P2_P3[patient_id_list[j]]:
                pred_P2_P3[patient_id_list[j]]=y_pred_temp2_ms3d[j]
            prob_accord[patient_id_list[j] ].append( y_pred_temp2_ms3d[j] )
        else:
            pred_P2_P3[patient_id_list[j]]=y_pred_temp2_ms3d[j]
            prob_accord[patient_id_list[j] ] = [ y_pred_temp2_ms3d[j] ]
            
    #print('#$!',len(pred_P2_P3), len(all_label_P2_P3))
    assert len(all_label_P2_P3) == len(pred_P2_P3)
    y_label_P2_P3, y_pred_P2_P3, y_id_P2_P3=[],[],[]
    for key in all_label_P2_P3.keys():
        pure_id = key[0:-2]
        y_label_P2_P3.append(all_label_P2_P3[key])
        y_id_P2_P3.append(key)
        if key not in pred_P2_P3:
            print('ERROR1', key)
        y_pred_P2_P3.append(pred_P2_P3[key])
        if key not in prob_accord:
            print('ERROR1', key)
        #p2p3_prob = '\t'.join(map(str, prob_accord[key])) if len(prob_accord[key]) > 1 else '-' + '\t' + str(prob_accord[key][0])
        #print(key, all_label_P2_P3[key], pred_P2_P3[key], p2p3_prob)
            
    return y_id_P2_P3, y_label_P2_P3, y_pred_P2_P3

def predict_patient(model, patient_label, lesion_data):
    label={}
    for idx in range(patient_label.shape[0]):
        if patient_label.shape[0] =="": continue
        case_id = patient_label[idx][1].split('/')[-1]
        case_id = case_id.split('.')[0]
        if case_id in label:
            print('ERROR1',case_id)
        else:
            label[case_id]=patient_label[idx][-1]
    
    batch_size = 16
    if len(lesion_data) == batch_size*int(len(lesion_data)/batch_size):
        num_batches = int(len(lesion_data)/batch_size)
    else:
        num_batches = int(len(lesion_data)/batch_size) + 1
    all_patient_id_list = []
    y_pred_ms3d, y_label_ms3d = [], []
    pred_P2_P3, all_label_P2_P3 = {},{}
    prob_accord = {}
    for i in range(num_batches):
        if i*batch_size+batch_size <= len(lesion_data):
            batch = lesion_data[i*batch_size: i*batch_size+batch_size]
        else:
            batch = lesion_data[i*batch_size: len(lesion_data)]
    x_data_batch = []
    patient_id_list = []
   
    for j in range(batch.shape[0]):
        tmp = np.load(batch[j][1])
        max_size = max(tmp.shape)
        min_size = min(tmp.shape)
        if min_size < width:
            tmp = padzero(tmp, width, height,depth)
        if max_size > width:
            tmp = resize_volume(tmp, width, height,depth)
        tmp = tmp / 255.0
        tmp[tmp>1]=1
        tmp[tmp<0]=0

        tmp = np.expand_dims(tmp, axis=0)
        x_data_batch.append(tmp)
        sample_id = os.path.basename(batch[j][0]).split('.np')[0]
        sample_id1 = sample_id.split('_P')[0] + sample_id.split('_P')[1][1:]
        patient_id_list.append(sample_id1)
        all_patient_id_list.append(sample_id)
        if sample_id1 in all_label_P2_P3:
            assert all_label_P2_P3[sample_id1] == int(batch[j][-1])
        else:
            all_label_P2_P3[sample_id1] = int(batch[j][-1])
        y_label_ms3d.append(int(batch[j][-1]))

    x_data_batch = np.array(x_data_batch).reshape((len(x_data_batch), width, height, depth))
    x_data_batch = np.expand_dims(x_data_batch, axis=-1)

    y_pred_temp2_ms3d = model.predict(x_data_batch).ravel()
    for j in range(batch.shape[0]):
        y_pred_ms3d.append(y_pred_temp2_ms3d[j])
        if patient_id_list[j] in pred_P2_P3:
            if y_pred_temp2_ms3d[j] > pred_P2_P3[patient_id_list[j]]:
                pred_P2_P3[patient_id_list[j]]=y_pred_temp2_ms3d[j]
            prob_accord[patient_id_list[j] ].append( y_pred_temp2_ms3d[j] )
        else:
            pred_P2_P3[patient_id_list[j]]=y_pred_temp2_ms3d[j]
            prob_accord[patient_id_list[j] ] = [ y_pred_temp2_ms3d[j] ]
            
    ########
    pred1={}
    for idx in range(len(all_patient_id_list) ):
        case_id = all_patient_id_list[idx][0:-2]
        if case_id not in pred1:
            pred1[case_id] = [float(y_pred_ms3d[idx])]
        else:
            pred1[case_id].append(float(y_pred_ms3d[idx]))

    pred={}
    #threshold=0.8
    threshold=0.78
    for key in pred1.keys():
        #print('#1', key, len(pred1[key]))
        if len(pred1[key])==1:
            #pred[key] = 1 if pred1[key][0] >= threshold else 0
            pred[key] = pred1[key][0]
        elif len(pred1[key])==2:
            prob = 0.6 * pred1[key][0] + 0.4* pred1[key][1] # old 7:3 old2: 0.55:0.45 old3: 0.45:0.55
            #pred[key] = 1 if prob >= threshold else 0
            pred[key] = prob
        elif len(pred1[key])==3:
            prob = 0.5 * pred1[key][0] + 0.3* pred1[key][1] + 0.2* pred1[key][2] # old 5:3:2 # old2: 0.4:0.3:0.3 old3:5:3:2
            #pred[key] = 1 if prob >= threshold else 0
            pred[key] = prob
        else:
            print('#1 ERROR', key)

    assert len(pred) == len(label)
    pred_arr, label_arr, pred_arr_binary, id_arr= [], [], [],[]
    exists_h = {}
    for key, value in pred.items():
        if key not in label:
            print('ERROR2',case_id)
            continue

        key_id = key.split('_P')[0]
        if key_id in exists_h:
            if value > pred_arr[exists_h[key_id]]:
                pred_arr[exists_h[key_id]] = value
                pred_arr_binary[exists_h[key_id]]= 1 if value >= threshold else 0
            assert label_arr[exists_h[key_id]] == label[key]
        else:
            #print('#1',type(label[key]))
            #print('#2',type(value))
            pred_arr.append(value)
            label_arr.append(label[key])
            id_arr.append(key_id)
            pred_bi = 1 if value >= threshold else 0
            pred_arr_binary.append(pred_bi)
            exists_h[key_id]=len(pred_arr_binary)-1
            #print(key, label[key], value,pred_bi)
    assert len(pred_arr)==len(label_arr)
    assert len(pred_arr)==len(pred_arr_binary)
    assert len(pred_arr) == len(id_arr)
    label_arr_np = np.array(label_arr).reshape((len(label_arr), 1))
    pred_arr_np = np.array(pred_arr).reshape((len(pred_arr), 1))
    ppred_arr_binary_np = np.array(pred_arr_binary).reshape((len(pred_arr_binary), 1))
    
    return id_arr, label_arr, pred_arr
    

