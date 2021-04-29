# Atrial fibrillation classification using deep learning algorithm in Internet of Things–based smart healthcare system

### Abstract
A partitioned deep convolutional neural network was proposed. The feature was learned 
continuously in the Internet of Things–based monitoring system and formed a high order
space.  
Sensitivity: True Positive rate  
Specificity: True Negative rate

### Introduction
3 big data V: Volume of data, Variety of data, Variability of data. The important aspects are as followed:
- ECG signal in Internet of Things (IoT) based smart healthcare system was obtained, and the dataset is 
  formed in lvm format file using myDAQ.
- Then, feature in ECG signal for training the 7-layer convolutional neural network (CNN) for training was computed. 
- Finally, the trained neural network is used for testing the remaining ECG dataset in the testing phase. 

The value of PR interval(increased), QRS complex interval(reduced), and presence of U wave(missing) is considered 
for identification of AFib ECG signal.
### Review of related work in ECG classification
### Materials used and methodology followed
7-layer CNN. The weight of network is initialized between 0 and 0.05. For training dataset, 20 batch size time series
data for 20 epochs are considered, and for testing interval, 100 epochs for every 10 iterations are considered.
### Experimental verification of deep CNN model
### Results and comparison of performance index for verification
### Conclusion