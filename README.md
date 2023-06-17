# ST3DCN
Spatio-Temporal 3D Convolution Network for diagnosis of hepatocellular carcinoma.

# System requirements
This project is based on Keras + Tensorflow, deployed on the platform with Centos whose version number is CentOS Linux release 7.7.1908 (Core). The hardware information of the server is as below.CPU: Intel(R) Xeon(R)  Gold 6146 CPU @ 3.20 GHz, 2 Thread(s) per core, 12 Core(s) per socket, 2 Socket(s)  ---> 48 CPU(s), GPU: 4 Tesla V100 32 GB, RAM Memory: 500 GB. 

The version of Keras (https://keras.io/getting_started/) is 2.11.0, the version of tensorflow-gpu (https://www.tensorflow.org/) is 2.11.0. 
To ensure the successful utilization of GPU computation power, one should install the correct version of GPU driver library first. As for us, the version of cudnn is 8.1.0.77 and the version of cudatoolkit is 11.2.2. One can install tensorflow-gpu and Keras under conda envirnment. Packages of numpy, scipy, panda, opencv, pillow, pydicom, etc, should be installed also. 

# Demon
```
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from ST3DCN_Utils import multi_scale_get_model_DCN
```
```
## Read file
image_mask_files = np.load('./data/image.npy', allow_pickle=True)
image_mask_files[:,:]

```
array([['./data/mask/ID_0848_P3.nii.gz', './data/image/ID_0848_P3', 1, 1,  0, 1],
       ['./data/mask/ID_0848_P2.nii.gz', './data/image/ID_0848_P2', 1, 1, 0, 1],
       ['./data/mask/QEH032_P3.nii.gz', './data/image/QEH032_P3', 0, 0, 0, 0],
       ['./data/mask/QEH032_P2.nii.gz', './data/image/QEH032_P2', 0, 0, 0, 0]], dtype=object)

```
draw(image_mask_files, 0,190,192)
```

<img width="972" alt="Screenshot 2023-06-17 at 5 51 54 PM" src="https://github.com/HKUMedicineLiverAI/ST3DCN/assets/136553001/31bbefd8-9c05-4edb-8558-854028976925">
<img width="972" alt="Screenshot 2023-06-17 at 5 51 44 PM" src="https://github.com/HKUMedicineLiverAI/ST3DCN/assets/136553001/469e4c2b-ead0-4d9f-b7e2-068b46c27911">

```
draw(image_mask_files, 2,286,287)
```
<img width="971" alt="Screenshot 2023-06-17 at 6 04 14 PM" src="https://github.com/HKUMedicineLiverAI/ST3DCN/assets/136553001/b4c9b38a-3cc2-4fed-8fe4-8a3df633bc8d">

```
width=70
height=70
depth = 70
factor=8
num_classes = 2
batch_size = 16
model = multi_scale_get_model_DCN(width=width, height=height, depth=depth, batch_size=batch_size, factor=factor, num_class=2)
model.load_weights(os.path.join('./weights','ST3DCN_Model.h5'))
image_mask_files1 = np.load('./data/data.npy', allow_pickle=True)
image_mask_files1[:,:]
```

```
## predict HCC or non-HCC for a patient
id_lis, label_lis, predict_lis = predict_patient(model, image_mask_files, image_mask_files1)
print('id','label','predict')
for i in range(len(id_lis)):
    print(id_lis[i],label_lis[i],predict_lis[i] )
    
```
1/1 [==============================] - 0s 31ms/step

ID_0848 1 0.8930234313011169

QEH032 0 0.1020175920566544

```
# predict HCC or non-HCC for a lesion
id_lis, label_lis, predict_lis = predict_lesion(model, image_mask_files, image_mask_files1)
print('id','label','predict')
for i in range(len(id_lis)):
    print(id_lis[i],label_lis[i],predict_lis[i] )
```
1/1 [==============================] - 0s 34ms/step

ID_0848_0 1 0.9005912

ID_0848_1 1 0.8816718

QEH032_0 0 0.0059577017

QEH032_1 0 0.23815411

QEH032_2 0 0.15053022

