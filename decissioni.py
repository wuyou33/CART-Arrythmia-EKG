import numpy as np
import os
import wfdb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets.base import Bunch
from sklearn.metrics import confusion_matrix
import csv
import json
from wfdb import processing as ps
import getpeak
from sklearn.ensemble import RandomForestClassifier
import scipy.signal
import xlsxwriter
import time
def load_my_HR_dataset(path):
    X=[]
    Y=[]
    samp = np.genfromtxt(path, delimiter=',', dtype=np.float32)
    samp2 = []
    j=1
    for j in range (len(samp)):
        if(j!=0):
            samp2.append(samp[j])
    for g in samp2:
        temp = [g[0],g[1],g[2],g[3],g[4],g[5],g[6]]
        X.append(temp)
        Y.append(g[7])
    Xa=np.array(X)
    Ya=np.array(Y)
    return Xa,Ya

def load_my_HR_dataset2(samp):
    X=[]
    for g in samp:
        temp = [g[0], g[1], g[2], g[3], g[4], g[5], g[6]]
        X.append(temp)
    Xa=np.array(X)
    return Xa

def make_mdl(data):
    start_time = time.time()
    dataset = data[0]
    target = data[1]
    clf = RandomForestClassifier(random_state= 100,n_estimators=100,criterion='entropy')
    clf = clf.fit(dataset, target)
    print("Model Fitted,\nTime execution to build model : ")
    print("--- %s seconds ---" % (time.time() - start_time))
    return clf

def test(dataset,clf):
    start_time = time.time()
    hasil = []
    for g in dataset:
        result = clf.predict(np.reshape(g, (1, -1)))
        hasil.append(result)
    print("Classification done,\nTime execution to build model : ")
    print("--- %s seconds ---" % (time.time() - start_time))
    return hasil

def test_result(dataset,clf):
    rs = test(dataset,clf)
    N = 0
    V = 0
    A = 0
    F = 0
    Q = 0
    U = 0
    overall=len(dataset)
    for w in rs:
        if (w == 0):
            N += 1
        elif (w == 1):
            V += 1
        elif (w == 2):
            A += 1
        elif (w == 3):
            F += 1
        elif (w == 4):
            Q += 1
        else:
            U += 1
    precentage = "Peak total = "+str(overall)+"\n" \
                 "Normal:"+str(N)+"\n" \
                 "VEB:"+str(V)+"\n" \
                 "SVEB:" +str(A)+ "\n" \
                 "FUSSION:" + str(F)+ "\n" \
                 "PACED:" +str(Q)+ "\n" \
                 "UNKNOWN:" +str(U)+ "" \


    print(precentage)
    return rs


def number_of_data(target):
    N = 0
    V = 0
    S = 0
    F = 0
    Q = 0
    U = 0
    for w in target:
        if (w == 0):
            N += 1
        elif (w == 1):
            V += 1
        elif (w == 2):
            S += 1
        elif (w == 3):
            F += 1
        elif (w == 4):
            Q += 1
        else:
            U += 1
    return(N,V,S,F,Q,U)

def accuracy(hasil,target):
    print('Akurasi model ='+str(round(accuracy_score(hasil, target)*100,3))+"%")
    df = confusion_matrix(target, hasil)
    print(str(df))
    # print(result)

def accuracy2(hasil,target):
    qz = number_of_data(target)
    N = qz[0]
    V = qz[1]
    S = qz[2]
    F = qz[3]
    Q = qz[4]
    U = qz[5]
    Nt = 0
    Vt = 0
    St = 0
    Ft = 0
    Qt = 0
    Ut = 0
    ovt = 0
    overall = len(hasil)
    print("COMBINED DATASET: \n"
          "Normal: "+str(N)+" \n"
          "VEB: "+str(V)+" \n"
          "SVEB: "+str(S)+" \n"
          "Fusion: "+str(F)+" \n"
          "Pacemaker: "+str(Q)+" \n"
          "Unknown: "+str(U))
    for j in range(len(hasil)):
        # print("Processing data : " + str(j + 1))
        if hasil[j] == target[j]:
            ovt += 1
            if (hasil[j] == 0):
                Nt += 1
            elif (hasil[j] == 1):
                Vt += 1
            elif (hasil[j] == 2):
                St += 1
            elif (hasil[j] == 3):
                Ft += 1
            elif (hasil[j] == 4):
                Qt += 1
            else:
                Ut += 1
    overall_acc = round((float(ovt / overall) * 100),2)
    Nacc = round((float(Nt / N) * 100),2)
    Vacc = round((float(Vt / V) * 100),2)
    Aacc = round((float(St / S) * 100),2)
    Facc = round((float(Ft / F) * 100),2)
    Qacc = round((float(Qt / Q) * 100),2)
    Uacc = round((float(Ut / U) * 100),2)
    df = confusion_matrix(target, hasil)
    print("Normal: \nDetected right:" + str(df[0][0]) + " Detected Wrong = " + str(
        df[0][1] + df[0][2] + df[0][3] + df[0][4] + df[0][5]) +
          "\nSVEB: \nDetected right:" + str(df[1][1]) + " Detected Wrong = " + str(
        df[1][0] + df[1][2] + df[1][3] + df[1][4] + df[1][5]) +
          "\nVEB: \nDetected right:" + str(df[2][2]) + " Detected Wrong = " + str(
        df[2][0] + df[2][1] + df[2][3] + df[2][4] + df[2][5]) +
          "\nFusion: \nDetected right:" + str(df[3][3]) + " Detected Wrong = " + str(
        df[3][0] + df[3][1] + df[3][2] + df[3][4] + df[3][5]) +
          "\nPACED: \nDetected right:" + str(df[4][4]) + " Detected Wrong = " + str(
        df[4][1] + df[4][2] + df[4][3] + df[4][5] + df[4][0]) +
          "\nUNK: \nDetected right:" + str(df[5][5]) + " Detected Wrong = " + str(
        df[5][1] + df[5][2] + df[5][3] + df[5][5] + df[5][0]))
    print(str(df))
    result = ("Accuracy:\n"
              "Overall: " + str(overall_acc) + "%\n"
              "Normal(N): " + str(Nacc) + "%\n"
              "Ventricular Ectopic Beat(VEB): " + str(
        Vacc) + "%\n"
                "Supraventricular Ectopic Beat(SVEB): " + str(Aacc) + "%\n"
                                                      "Fusion(F): " + str(Facc) + "%\n"
                                                                                   "Paced using Pacemaker(Q): " + str(
        Qacc) + "%\n"
                "Unknown/Non-Beat(U): " + str(Uacc) + "%")
    print(result)
def main():
    trainer = load_my_HR_dataset(r"..\EKGReader\full.csv")
    clf = make_mdl(trainer)
    chs = "YES"
    while (chs != "NO"):
                print("\1.TESTING tree\n2.TESTING TREE WITH EKG DATASET\nSELECT MENU")
                answer = input()
                if (answer == "1"):
                    incart_all = ['I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10',
                                  'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20',
                                  'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30',
                                  'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20',
                                  'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40',
                                  'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50',
                                  'I51', 'I52', 'I53', 'I54', 'I55', 'I56', 'I57', 'I58', 'I59', 'I60',
                                  'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70',
                                  'I71', 'I72', 'I73', 'I74', 'I75'
                                  ]
                    ans2 = "YES"
                    while ans2 != "NO":
                        w = input("1.Test by data per data\n2.Test by testing set\n3.Test with all dataset\nSelect")
                        if (w == "1"):
                            for g in incart_all:
                                dir_data = r"..\ECG\EKGReader"
                                sample = "\\" + g
                                types = ".csv"
                                files = dir_data + sample + types
                                tester = load_my_HR_dataset(files)
                                tester_data = tester[0]
                                tester_class = tester[1]
                                hasil = test(tester_data, clf)
                                accuracy(hasil, tester_class)
                                print("RECORD TESTING : " + g)
                            ans2 = input("Try another test? YES/NO")
                            if (ans2 == 0):
                                chs = input("Exit to main menu? YES/NO")
                                if (chs == "NO"):
                                    exit()
                        if (w == '2'):
                            tester = load_my_HR_dataset(r"..\EKGReader\test.csv")
                            tester_data = tester[0]
                            tester_class = tester[1]
                            hasil = test(tester_data, clf)
                            accuracy2(hasil, tester_class)
                            ans2 = input("Try another test? YES/NO")
                            if (ans2 == 0):
                                chs = input("Exit to main menu? YES/NO")
                                if (chs == "NO"):
                                    exit()
                        if (w == '3'):
                            tester = load_my_HR_dataset(r"..\EKGReader\allin.csv")
                            tester_data = tester[0]
                            tester_class = tester[1]
                            hasil = test(tester_data, clf)
                            accuracy2(hasil, tester_class)
                            ans2 = input("Try another test? YES/NO")
                            if (ans2 == 0):
                                chs = input("Exit to main menu? YES/NO")
                                if (chs == "NO"):
                                    exit()
                elif (answer == "2"):
                    ecg=[]

                    datas = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
                    for w in range(len(datas)):
                        print("Sample : "+str(datas[w]))
                        # patient = input("Masukkan data pasien")
                        patient = str(datas[w])
                        filename = "\\" + patient
                        dir = "..\EKGReader\Data\json"
                        typed = ".json"
                        path = dir + filename + typed
                        signal = getpeak.load_tester(path)
                        for g in range(5000):
                            ecg.append(signal[g])
                        #rate = samplerate sinyal
                        rate = 100
                        # Band pass batas bawah
                        lowfreq = 5.0
                        # Band pass batas atas
                        highfreq = 15.0
                        lopass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
                        hipass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
                        # TODO: Bandpass filter, modifikasi disini
                        ecg_lo = scipy.signal.filtfilt(*lopass, x=ecg)
                        ecg_bd = scipy.signal.filtfilt(*hipass, x=ecg_lo)

                        # Square (=signal power) turunan sinyal, dan dikuadratkan
                        dfecg = np.diff(ecg_bd)
                        decg_pwr = np.square(dfecg)
                        signew = decg_pwr

                        pkr = getpeak.detect_beats(signal)
                        features = getpeak.makefeat(signew, pkr)
                        fullist = load_my_HR_dataset2(features)
                        rs = test_result(fullist, clf)
                        ANN = []
                        for k in rs:
                            ks = int(k)
                            if ks == 0:
                                ANN.append('N')
                            elif ks == 1:
                                ANN.append('V')
                            elif ks == 2:
                                ANN.append('A')
                            elif ks == 3:
                                ANN.append('F')
                            elif ks == 4:
                                ANN.append('P')
                            else:
                                ANN.append('U')
                        for k in range(len(features)):
                            features[k].append(ANN[k])

                        x = []  # length of signal
                        x2 = []  # length index of signal
                        y1 = []  # amplitude of raw signal
                        y2 = []  # amplitude of r-peak
                        y3 = []  # amplitude of preprocessed signal
                        y4 = []  # peak of preprocessed signal
                        nnew = []
                        for g in range(0,2000):
                            x.append(g)
                        for h in range(0,2000):
                            y1.append(signew[h])
                        for h in range(0,2000):
                            y3.append(signal[h])
                        for i in range(50):
                            x2.append(pkr[i])
                        for k in range(50):
                            y2.append(signal[x2[k]])
                        for w in range(50):
                            y4.append(1)
                        for m in range(50):
                            nnew.append(ANN[m])
                        print(len(x),len(x2),len(y1),len(y2),len(y3),len(y4))
                        from matplotlib import pyplot as plt
                        fig, ax = plt.subplots()
                        ax.scatter(x2, y4, c='orange' ,s=3)
                        plt.xlabel('time (n/100 .s)')
                        plt.ylabel('Amplitude')
                        plt.title('ECG record ' + patient)
                        for i, txt in enumerate(nnew):
                            ax.annotate(txt, (x2[i], y4[i]))
                            plt.axvline(x=x2[i], linewidth=0.5, c='red')
                        plt.plot(x, y3, linewidth=0.3, c='black',label='Raw')
                        plt.plot(x, y1, linewidth=0.8, c='blue',label='Preprocessed')
                        plt.axis([1500, 2000, -1.5,1.5])
                        fig.show()
                        fig.savefig(r'..\EKGReader\Data\json' + '\\' + str(
                            patient) + "plotted.png")  # save the figure to file
                        pdr = pd.DataFrame(features, columns=["Amplitude", "1 Back", "1 Forward", "RR", "HR back", "HR",
                                                              "HR After", "TYPE"])

                        status = 'idle'
                        if(int(patient)/2!=0):
                            status = ' Duduk'
                        else: status='Jalan'
                        writer = pd.ExcelWriter(r'..\EKGReader\\Data\json\\''features ' + str(patient) + " " + status+'.xlsx', engine = 'xlsxwriter')
                        pdr.to_excel(writer, sheet_name='features '+ str(patient) +" "+ status)
                        writer.save()

                    chs = input("Exit to main menu? YES/NO")
                    if (chs == "NO"):
                        exit()
                elif (answer == '3'):
                    if (chs == "NO"):
                        exit()


