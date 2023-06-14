# ST3DCN
Spatio-Temporal 3D Convolution Network for diagnosis of hepatocellular carcinoma.

# System requirements and operating system including version numbers
This project is based on Keras + Tensorflow, deployed on the platform with Centos whose version number is CentOS Linux release 7.7.1908 (Core). The hardware information of the server is as below.CPU: Intel(R) Xeon(R)  Gold 6146 CPU @ 3.20 GHz, 2 Thread(s) per core, 12 Core(s) per socket, 2 Socket(s)  ---> 48 CPU(s), GPU: 4 Tesla V100 32 GB, RAM Memory: 500 GB. Although the proposed model is trained on the HPC server, one can reduce the hardware requirement by reducing the resolution of input CT image volume. 

# The version of Keras (https://keras.io/getting_started/) is 2.2.4, the version of tensorflow-gpu (https://www.tensorflow.org/) is 1.15.0. 
To ensure the successful utilization of GPU computation power, one should install the correct version of GPU driver library first. As for us, the version of cudnn is 7.6.0 and the version of cudatoolkit is 10.1.168. Second, one can install tensorflow-gpu and Keras. In the following, one should install numpy, scipy, panda, opencv, pillow, pydicom, etc. 

# Installation Guide: Instruction + Typical Install time on a normal desktop computer
(1) Instrcution has desribed above, which can be summaried by: cudnn, cudatoolkit --> tensorflow-gpu, keras ---> other required libraries like numpy, scipy, and so on.

(2) Generally, the installation can be completed within two or four hours when the network connection has a good quality.

For easy installation, one can watch this video for more detailed information via 
https://www.youtube.com/watch?v=OEFKlRSd8Ic&ab_channel=JeffHeaton.

# Demon
(1) Instruction to run on data: 

For public or private liver CT volumes, one can first use the pretrained liver detection model pre-trained on the public data sets LITS (https://competitions.codalab.org/competitions/17094) to capture all the slices that contain liver. Then one can use the registration code to transform the liver CT scans to 3D-shape volumes and then resize to (256, 256, 128), where the first two 256 denote the number of pixels along with the width and height directions, and 128 denotes the number of slices. Third, one can train the model on the resized CT volumes. Once the model training is completed, one can conduct evaluation by loading the well-trained model weights.  

(2) Expected output:

The outputs of the proposed model depends on the avaiability of mask files, i.e., the lesion contouring information. If the lesion contouring files are avaiable, one can run the model at the observation/lesion level. The output of model includes the probability of a patient/case being diagnoised as HCC/NonHCC (the classification task) and the prediction of lesion area detection (the segmentation task). It is noteworth that the output of lesion area detection only indicates the existence of lesions, not responsible for HCC/Non-HCC classification.

If the lesion contouring files is not available, one can run the model at the patient leve. In this case, the output is the probability of a patient/case being diagnoised as HCC/NonHCC.

(3) Expected run time for demo on a "normal" desktop computer

The model training on our server takes about 120 hours. For a general desktop computer, if there are one or two GPU cards, it will take about at least 240 hours after resizing the CT images to run the model normally on general desktop computer. Although the model training costs days, the model evaluation can be done within several minutes. 

# Instructions for use

(1) How to run the software on your data

(2) (Optinal) repreoduction instructions
