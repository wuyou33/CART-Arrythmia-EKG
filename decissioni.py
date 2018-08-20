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
    dataset = data[0]
    target = data[1]
    clf = RandomForestClassifier(random_state= 1000 ,n_estimators=1000, criterion='gini')
    clf = clf.fit(dataset, target)
    print("Model Fitted")
    return clf

def test(dataset,clf):
    hasil = []
    for g in dataset:
        result = clf.predict(np.reshape(g, (1, -1)))
        hasil.append(result)
    print("Classification done")
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
    overall_acc = float(ovt / overall) * 100
    Nacc = float(Nt / N) * 100
    Vacc = float(Vt / V) * 100
    Aacc = float(St / S) * 100
    Facc = float(Ft / F) * 100
    Qacc = float(Qt / Q) * 100
    Uacc = float(Ut / U) * 100
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
    trainer = load_my_HR_dataset(r"E:\ECG\EKGReader\full.csv")
    clf = make_mdl(trainer)
    trainer2 = load_my_HR_dataset(r"E:\ECG\EKGReader\allin.csv")
    clt = make_mdl(trainer2)
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
                                dir_data = r"E:\ECG\EKGReader"
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
                            tester = load_my_HR_dataset(r"E:\ECG\EKGReader\test.csv")
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
                            tester = load_my_HR_dataset(r"E:\ECG\EKGReader\allin.csv")
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

                    patient = input("Masukkan data pasien")
                    filename = "\\" + patient
                    dir = r"E:\ECG\EKGReader\Data\json"
                    typed = ".json"
                    path = dir + filename + typed
                    signal = getpeak.load_tester(path)
                    pkr = getpeak.makefeat2(signal)
                    features  = getpeak.makefeat(signal,pkr)
                    fullist = load_my_HR_dataset2(features)
                    rs = test_result(fullist, clt)
                    ANN = []
                    for k in rs:
                        ks = int(k)
                        if ks==0: ANN.append('N')
                        elif ks == 1: ANN.append('V')
                        elif ks == 2: ANN.append('A')
                        elif ks == 3: ANN.append('F')
                        elif ks == 4: ANN.append('P')
                        else: ANN.append('U')
                    print(ANN)
                    #wfdb.plot_items(signal=signal,annotation=[ann.sample], title=patient, time_units='seconds',figsize=(10, 4), ecg_grids='all')
                    chs = input("Exit to main menu? YES/NO")
                    if (chs == "NO"):
                        exit()
                elif (answer == '3'):

                    if (chs == "NO"):
                        exit()


