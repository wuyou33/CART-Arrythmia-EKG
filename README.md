# CART-Arrythmia-EKG
EKG Arrythmia classficiation using CART method
## Method
This arrythmia classification using Classification and Regression Tree method , using dataset from MIT-BIH Arrythmia dataset. The datasets are divided into two parts, for training and testing purpose.
## Patient dataset
Patient dataset are taken from Bitalino(R) micro-EKG machine and extracted into .json file.
## Features
The features used in classification is:

    1.QRS-complex peak
    2.QRS-complex amplitude 1 behind
    3.QRS-Complex amplitude 1 forward
    4.R-R interval (distance between two nearest QRS peak)
    5.Heart Rate 1 step behind
    6.Heart Rate (speed in bpm)
    7.Heart Rate 1 step ahead
    8.Heart Rate variance in one record

## Preprocessing and feature extraction
Patient raw data are preprocessed using WFDB http://wfdb.readthedocs.io/ for extract QRS location and peak. Patient dataset using samplerate 100(I have using 360,320 and 1000 samplerate , and the result are very bad).
Training and testing dataset are preprocessed using WFDB for extract the annotation, QRS peaks.
Dataset collected from Physionet are MIT-BIH dataset and INCART-St Petersburg 12-lead records.
## Class code
    1.N = Normal , for programming purpose coded as 0
    2.V = PVC (Premature Ventricular Contraction), extra beat (This is the sign of arrythmia), for programming purpose coded as 1
    3.A = APC (Atrial Premature Contraction), intensive and strong beat (This is the sign of arrythmia), for programming purpose coded as 2
    4.F = Fussion beat, Dressler beat, sign of Ventricular Tachycardia, for programming purpose coded as 3
    5.P = Paced, heart paced using artificial pacemaker, for programming purpose coded as 4
    6.U = Unknown, non-beat., for programming purpose coded as 5
  
## Bibliography
Amri, M.F., Rizqyawan, M.I. and Turnip, A., 2016. ECG signal processing using offline-wavelet transform method based on ECG-IoT device. 2016 3rd International Conference on Information Technology, Computer, and Electrical Engineering (ICITACEE), pp.1–6.

Luz, E.J. da S., Schwartz, W.R., Cámara-Chávez, G. and Menotti, D., 2016. ECG-based heartbeat classification for arrhythmia detection: A survey. Computer Methods and Programs in Biomedicine, [online] 127, pp.144–164. Available at: <http://dx.doi.org/10.1016/j.cmpb.2015.12.008>.

## Library Used
Sci-Kit Learn
Numpy
WFDB
