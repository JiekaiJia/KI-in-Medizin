# Classification of ECG signals using Machine Learning Techniques: A Survey

### Introduction
Major issues in ECG classification are lack of standardization of ECG features, 
variability amongst the ECG features, **individuality of the ECG patterns?**, nonexistence of optimal classification
rules for ECG classification, and variability in ECG waveforms of patients.

### Background Knowledge
Preprocessing, feature extraction, normalization, and classification are main sequential steps of ECG classification.
The utilization of **pattern classifier techniques?** can improve the new patients ECG arrhythmia diagnoses.
- **Preprocessing**  
  necessary for removing the noises(e.g. baseline wander noise). For noise removal, techniques such as low pass linear 
  phase filter, linear phase high pass filter etc. are used. For baseline adjustment, techniques such as median filter, 
  linear phase high pass filter, mean median filter etc. are used.
- **Feature extraction**  
  Feature extraction techniques used by researchers are Discrete Wavelet Transform (DWT), Continuous Wavelet Transform
  (CWT), Discrete Cosine Transform (DCT), S-Transform (ST), Discrete Fourier transform (DFT), Principal Component 
  Analysis (PCA), Daubechies wavelet (Db4), Pan-Tompkins algorithm, Independent Component Analysis (ICA) etc.
- **Normalization**  
  For normalization of features, techniques such as Z-score and Unity Standard Deviation (SD) are used.
- **Classification**  
  Classification techniques used are Multilayer Perceptron Neural Network(MLPNN), Fuzzy C-Means clustering (FCM), 
  Feed forward neuro-fuzzy, ID3 decision tree, Support Vector Machine(SVM), Quantum Neural Network (QNN), Radial Basis 
  Function Neural Network (RBFNN), Type2 Fuzzy Clustering Neural Network (T2FCNN) and Probabilistic Neural Network
  (PNN) classifier etc.
  
### Issues ECG Classification
- **individuality of the ECG patterns**
  Individuality of the ECG patterns: Individuality of the ECG pattern refers to the likelihood of intraclass similarity 
  and interclass variability of testing patterns observed in ECG data. It shows up to what extent the ECG patterns are 
  scalable in sufficiently larger dataset.
  
### Survey of ECG Classification
### ECG Classification
1. **ECG Database**
2. **Feature extraction technique**
3. **Classification of ECG using neural network**