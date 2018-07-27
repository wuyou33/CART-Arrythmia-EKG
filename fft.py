import wfdb
import pandas as pd
import numpy as np
import random as rd
from wfdb import processing as ps


def dataset_making(sig):
    signal = []
    ann = wfdb.rdann('mitdb/' + str(sig), 'atr')
    sig, fields = wfdb.rdsamp('mitdb/' + str(sig), channels=[1])
    samplerate = ann.fs
    #print(samplerate)
    cd = []
    for w in range(len(ann.sample)):
        types = ann.symbol[w]
        array = [ann.sample[w], types]
        cd.append(array)
    for k in range(len(sig)):
        sgt = sig[k][0]
        temps = [k, sgt]
        signal.append(temps)
    A = cd
    B = signal
    tot = 0
    # print(B)
    # for w in B:
    #     tot+=w[1]
    # avg = tot/len(B)
    # for x in B:
    #     x[1] = np.square((x[1]-avg))
    B_Dict = {b[0]: b for b in B}
    array_new = [[B_Dict.get(a[0])[0], B_Dict.get(a[0])[1], a[1]] for a in A if B_Dict.get(a[0])]
    dataset = []
    for j in range(len(array_new)):
        for k in range(len(array_new[j])):
            amplitude = array_new[j][1]
            d1 = array_new[j][0]
            if (j < (len(array_new) - 1)):
                d2 = array_new[j + 1][0]
                distance = d2 - d1
                RR = round(((distance) / samplerate), 3)
                HR = (60 / RR)
            else:
                RR = 0
                HR = 0
            # if(j==0):
            #     dob=0
            # else:
            #     dob= round(np.absolute(array_new[j][1]-array_new[j-1][1]),3)
            # if(j==len(array_new)-1):
            #     doa= 0
            # else:
            #     doa=round(np.absolute(array_new[j+1][1]-array_new[j-1][1]),3)

            class_data = (array_new[j][2])
            temp = [amplitude, RR, HR, class_data]
            k + +1
        dataset.append(temp)
    return dataset


def writen(set):
    pdr = pd.DataFrame(set, columns=["Amplitude", "Var Amp", "RR", "Var RR", "HR", "Var HR", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\test.csv', index=False)

def writen2(set):
    pdr = pd.DataFrame(set, columns=["Amplitude", "Var Amp", "RR", "Var RR", "HR", "Var HR", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\fullx.csv', index=False)

def writenw(set):
    pdr = pd.DataFrame(set, columns=["Amplitude", "Var Amp", "RR", "Var RR", "HR", "Var HR", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\weight.csv', index=False)


def writent(set):
    pdr = pd.DataFrame(set, columns=["Amplitude", "Var Amp", "RR", "Var RR", "HR", "Var HR", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\full.csv', index=False)


def featuring(len_set):
    feature = []
    for g in range(len(len_set)):
        print("Processing data " + str(g + 1))
        arv = (dataset_making(sig[g]))
        for k in range(len(arv)):
            if (k == 0 or k == (len(arv) - 1)):
                continue
            else:
                feature.append(arv[k])
    for j in range(len(feature)):
        feature[j][3] = dtypes(feature[j][3])
    return feature


def dtypes(types):
    if ((types == "N") or (types == "R") or (types == "L") or (types == "e") or (types == "j")):
        types = 0  # "N" #
    elif ((types == "V") or (types == "E")):
        types = 1  # "V" #
    elif ((types == "A") or (types == "J") or (types == "a") or (types == "S")):
        types = 2  # "S" #
    elif (types == "F"):
        types = 3  # "F" #
    elif ((types == "Q") or (types == "f") or (types == "/")):
        types = 4  # "Q" #
    else:
        types = 5  # "else"
    return types


sig = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123,
       124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230,
       231, 232, 233, 234]
ds2 = [100, 103, 105, 111, 113, 117, 121, 123,
       200, 202, 210, 212, 213, 214, 219, 221,
       222, 228, 231, 232, 233, 234]

ds1 = [101, 106, 108, 109, 112, 114, 115, 116, 118,
       119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223,
       230]

ds2 = [100, 103, 105, 11, 113, 117, 121, 123,
       200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233
    , 234]


def statistical_feature(feat):
    N = []
    V = []
    A = []
    F = []
    Q = []
    U = []
    for w in feat:
        if (w[3] == 0):
            N.append(w)
        elif (w[3] == 1):
            V.append(w)
        elif (w[3] == 2):
            A.append(w)
        elif (w[3] == 3):
            F.append(w)
        elif (w[3] == 4):
            Q.append(w)
        else:
            U.append(w)
    print("Normal Data Set")
    print(len(N))
    print(len(V))
    print(len(A))
    print(len(F))
    print(len(Q))
    print(len(U))
    stat = []
    amp = []
    rr = []
    hr = []
    for w in range(len(feat)):
        amp.append(feat[w][0])
        rr.append(feat[w][1])
        hr.append(feat[w][2])
    at = 0
    rt = 0
    ht = 0
    l = len(amp)
    for j in range(l):
        at += amp[j]
        ht += hr[j]
        rt += rr[j]
    ava = float(at / l)
    avr = float(rt / l)
    avh = float(ht / l)
    for j in range(l):
        amp[j] = round((np.square(amp[j] - ava)), 3)
        rr[j] = round((np.square(rr[j] - avr)), 3)
        hr[j] = round((np.square(hr[j] - avh)), 3)
    for k in range(len(feat)):
        stat.append([feat[k][0], (amp[k]), feat[k][1], (rr[k]), feat[k][2], (hr[k]), feat[k][3]])
    return stat


# def setunique(items):
#     seen = set()
#     for item in items:
#         item = tuple(item)
#         if item not in seen:
#             yield item
#             seen.add(item)
#     return seen
#
#
# def feature_training(sist):
#     N = []
#     V = []
#     A = []
#     F = []
#     Q = []
#     U = []
#     dataset = []
#     ln = 700
#     for w in sist:
#         if (w[5] == 0):
#             N.append(w)
#         elif (w[5] == 1):
#             V.append(w)
#         elif (w[5] == 2):
#             A.append(w)
#         elif (w[5] == 3):
#             F.append(w)
#         elif (w[5] == 4):
#             Q.append(w)
#         else:
#             U.append(w)
#
#     for j in range(ln):
#         dataset.append(N[j])
#     for j in range(ln):
#         dataset.append(V[j])
#     for j in range(ln):
#         dataset.append(A[j])
#     for j in range(ln):
#         dataset.append(F[j])
#     for j in range(ln):
#         dataset.append(Q[j])
#     for j in range(ln):
#         dataset.append(U[j])
#     return dataset
#
#
# def feature_weight(sist):
#     N = []
#     V = []
#     A = []
#     F = []
#     Q = []
#     U = []
#     dataset = []
#     ln = 1
#     for w in sist:
#         if (w[5] == 0):
#             N.append(w)
#         elif (w[5] == 1):
#             V.append(w)
#         elif (w[5] == 2):
#             A.append(w)
#         elif (w[5] == 3):
#             F.append(w)
#         elif (w[5] == 4):
#             Q.append(w)
#         else:
#             U.append(w)
#     print("Reduced Data Set")
#     print(len(N))
#     print(len(V))
#     print(len(A))
#     print(len(F))
#     print(len(Q))
#     print(len(U))
#     for j in range(ln):
#         dataset.append(N[rd.randint(0, ((len(N)) - 1))])
#     for j in range(ln):
#         dataset.append(V[rd.randint(0, ((len(V)) - 1))])
#     for j in range(ln):
#         dataset.append(A[rd.randint(0, ((len(A)) - 1))])
#     for j in range(ln):
#         dataset.append(F[rd.randint(0, ((len(F)) - 1))])
#     for j in range(ln):
#         dataset.append(Q[rd.randint(0, ((len(Q)) - 1))])
#     for j in range(ln):
#         dataset.append(U[rd.randint(0, ((len(U)) - 1))])
#     return dataset

# fw=featuring(sig)
# fw2=featuring(ds2)
# for w in fw:
#     print(w)
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
feature1 = featuring(ds1)
f1 = statistical_feature(feature1)
writen(f1)
feature = featuring(ds2)
f2 = statistical_feature(feature)
writent(f2)
feature3 = featuring(sig)
f3 = statistical_feature(feature3)
writen2(f3)
# feature2 = featuring(ds2)


# for j in feature:
#     print (j)
# for j in ft:
#     print (j)
# f3=feature_weight(feature2)

# writenw(f3)
