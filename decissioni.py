import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.datasets.base import Bunch
from sklearn import tree
from sklearn.metrics import confusion_matrix
import csv
import json
import pandas as pd
import numpy as np
from wfdb import processing as ps
import getpeak

def load_my_HR_dataset(path):
    X=[]
    Y=[]
    samp = np.genfromtxt(path, delimiter=',', dtype=np.float32)
    for g in samp:
        temp = [g[0],g[1],g[2],g[3],g[4],g[5]]
        X.append(temp)
        Y.append(g[6])
    Xa=np.array(X)
    Ya=np.array(Y)
    return Xa,Ya

def load_my_HR_dataset2(path):
    X=[]
    samp = np.genfromtxt(path, delimiter=',', dtype=np.float32)
    for g in samp:
        temp = [g[0],g[1],g[2],g[3],g[4],g[5]]
        X.append(temp)
        #Y.append(g[6])
    Xa=np.array(X)
    return Xa

def make_mdl(data):
    dataset = data[0]
    target = data[1]
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(dataset, target)
    print(clf)
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
    hasil=""
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
            hasil+"Normal\n"
        elif (w == 1):
            V += 1
            hasil + "PVC\n"
        elif (w == 2):
            A += 1
            hasil + "APC\n"
        elif (w == 3):
            F += 1
            hasil + "Fussion\n"
        elif (w == 4):
            Q += 1
            hasil + "Paced\n"
        else:
            U += 1
            hasil + "Unknown\n"
    precentage = "Normal:"+str(round(N/overall,3)*100)+"%\n" \
                 "PVC:"+str(round(V/overall,3)*100)+"%\n" \
                 "APC:" + str(round(A / overall, 3)*100) + "%\n" \
                 "FUSSION:" + str(round(F / overall, 3)*100) + "%\n" \
                 "PACED:" + str(round(Q / overall, 3)*100) + "%\n" \
                 "UNKNOWN:" + str(round(U / overall, 3)*100) + "%\n"
    hasil=hasil+precentage
    print(hasil)



def number_of_data(target):
    N = 0
    V = 0
    A = 0
    F = 0
    Q = 0
    U = 0
    for w in target:
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
    return(N,V,A,F,Q,U)

def accuracy(hasil,target):
    qz = number_of_data(target)
    N = qz[0]
    V = qz[1]
    A = qz[2]
    F = qz[3]
    Q = qz[4]
    U = qz[5]
    Nt = 0
    Vt = 0
    At = 0
    Ft = 0
    Qt = 0
    Ut = 0
    ovt = 0
    overall = len(hasil)
    for j in range(len(hasil)):
        print("Processing data : " + str(j + 1))
        if hasil[j] == target[j]:
            ovt += 1
            if (hasil[j] == 0):
                Nt += 1
            elif (hasil[j] == 1):
                Vt += 1
            elif (hasil[j] == 2):
                At += 1
            elif (hasil[j] == 3):
                Ft += 1
            elif (hasil[j] == 4):
                Qt += 1
            else:
                Ut += 1

    overall_acc = float(ovt / overall) * 100
    Nacc = float(Nt / N) * 100
    Vacc = float(Vt / V) * 100
    Aacc = float(At / A) * 100
    Facc = float(Ft / F) * 100
    Qacc = float(Qt / Q) * 100
    Uacc = float(Ut / U) * 100
    result = ("Accuracy:\n"
              "Overall: " + str(overall_acc) + "%\n"
              "Normal(N): " + str(Nacc) + "%\n"
              "Premature Ventribular Contraction(V): " + str(
        Vacc) + "%\n"
                "Atrial Premature(A): " + str(Aacc) + "%\n"
                                                      "Fussion(F): " + str(Facc) + "%\n"
                                                                                   "Paced using Pacemaker(Q): " + str(
        Qacc) + "%\n"
                "Unknown/Non-Beat(U): " + str(Uacc) + "%")
    print(str(confusion_matrix(target, hasil)))
    print(result)




def main():

    trainer = load_my_HR_dataset(r"E:\ECG\EKGReader\full.csv")
    tester = load_my_HR_dataset(r"E:\ECG\EKGReader\test.csv")
    #fullist = load_my_HR_dataset2(r"E:\ECG\EKGReader\samp1.csv")

    tester_data = tester[0]
    tester_class = tester[1]
    clf = make_mdl(trainer)
    chs = "YES"
    while (chs != "NO"):
        print("1.TRAINING Tree\n2.TESTING tree\n3.TESTING TREE WITH EKG DATASET\nSELECT MENU")
        answer = input()
        if (answer == "1"):
            clf = make_mdl(trainer)
            chs = input("Exit to main menu? YES/NO")
            if (chs == "NO"):
                exit()
        elif (answer == "2"):
            hasil = test(tester_data,clf)
            accuracy(hasil,tester_class)
            chs = input("Exit to main menu? YES/NO")
            if (chs == "NO"):
                exit()
        elif (answer == "3"):
            fileset = ["samp1", "samp2", "samp3", "samp4", "samp5"]
            for w in fileset:
                filename = "\\" + str(w)
                dir = r"E:\ECG\EKGReader\Data\json"
                type = ".json"
                path = dir + filename + type
                dirout = r"E:\ECG\EKGReader"
                typeout = ".csv"
                dirouts = dirout+filename+typeout
                signal = getpeak.load_tester(path)
                featureist = getpeak.makefeat(signal)
                pdr = pd.DataFrame(featureist, columns=["Amplitude", "Var Amp", "RR", "Var RR", "HR", "Var HR"])
                pdr.to_csv(dirouts, index=False)

            for k in fileset:
                filename = "\\" + str(k)
                dir = r"E:\ECG\EKGReader"
                type = ".csv"
                path = dir + filename + type
                fullist = load_my_HR_dataset2(path)
                stats = []
                for w in range(len(fullist)):
                    if (w != 0):
                        stats.append(fullist[w])
                stats = np.asarray(stats)
                test_result(stats, clf)

            chs = input("Exit to main menu? YES/NO")
            if (chs == "NO"):
                exit()