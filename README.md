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
Ashley, E. a and Niebauer, J., 2004. Cardiology Explained. Remedica.
Breiman, L., 2001. Random Forest. pp.1–33.
Chen, J. and Itoh, S., 1998. A Wavelet Transform-Based ECG Compression. 45(12), pp.1414–1419.
Clifford, G., 2006. Advanced Method and Tools for ECG Data Analysis.
Goldberger, A.L., Amaral, L.A.N., Glass, L., Hausdorff, J.M., Ivanov, P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C. and Stanley, H.E., 2014. Current Perspective PhysioBank, PhysioToolkit, and PhysioNet Components of a New Research Resource for Complex Physiologic Signal.
Iqbal, U., Wah, T.Y., Habib Ur Rehman, M. and Mastoi, Q.U.A., 2018. Usage of Model Driven Environment for the Classification of ECG features: A Systematic Review. IEEE Access, 6, pp.23120–23136.
Köhler, B.U., Hennig, C. and Orglmeister, R., 2002. The principles of software QRS detection. IEEE Engineering in Medicine and Biology Magazine, 21(1), pp.42–57.
Li, W. and Li, J., 2017. Local Deep Field for Electrocardiogram Beat Classification. IEEE Sensors Journal, 18(4), pp.1656–1664.
Luz, E.J. da S., Schwartz, W.R., Cámara-Chávez, G. and Menotti, D., 2016. ECG-based heartbeat classification for arrhythmia detection: A survey. Computer Methods and Programs in Biomedicine, [online] 127, pp.144–164. Tersedia di: <http://dx.doi.org/10.1016/j.cmpb.2015.12.008>.
Nag, P., Mondal, S., Ahmed, F., More, A. and Raihan, M., 2017. A simple acute myocardial infarction (Heart Attack) prediction system using clinical data and data mining techniques. 2017 20th International Conference of Computer and Information Technology (ICCIT), pp.1–6.
Spach, M.S. and Kootsey, J.M., 1983. The nature of electrical propagation in cardiac muscle. The American journal of physiology, 244(1), pp.H3–H22.
Texas Heart Institute, 2018. Categories of Arrhythmias. [online] Tersedia di: <https://www.texasheart.org/heart-health/heart-information-center/topics/categories-of-arrhythmias/> [Diakses 4 Agustus 2018].



## Library Used
Sci-Kit Learn
Numpy
WFDB
