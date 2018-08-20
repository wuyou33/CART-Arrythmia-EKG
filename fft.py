import wfdb
import pandas as pd
import numpy as np

def dataset_making(sig):
    signal = []
    dataset = []
    ann = wfdb.rdann('INCART/' + sig, 'atr')
    sig, fields = wfdb.rdsamp('INCART/' + sig, channels=[7])
    samplerate = ann.fs
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
    B_Dict = {b[0]: b for b in B}
    array_new = [[B_Dict.get(a[0])[0], B_Dict.get(a[0])[1], a[1]] for a in A if B_Dict.get(a[0])]

    for j in range(len(array_new)):
        for k in range(len(array_new[j])):
            amplitude = array_new[j][1]
            d1 = array_new[j][0]
            if(j==0 or j!=(len(array_new) - 1)):
                d2 = array_new[j+1][0]
                distance = d2 - d1
                RR = (distance/samplerate)
                HR=  int(60/RR)
            else:
                RR=0
                HR=0
            if(j==0):
                toward = array_new[j+1][1]
                toback = 0
            if(j==(len(array_new) - 1)):
                toward = 0
                toback =array_new[j - 1][1]
            else:
                toward = array_new[j + 1][1]
                toback = array_new[j - 1][1]
            class_data = (array_new[j][2])
            # temp = [amplitude, toback, toward, RR, HR, class_data]
            temp = [amplitude,toback,toward,RR, HR, class_data]
        dataset.append(temp)
    dtn = []
    amp =[]
    bk = []
    fr = []
    rrt = []
    hr = []
    cld = []
    hrb=[]
    hra=[]
    for h in range(len(dataset)):
        amp.append(dataset[h][0])
        bk.append(dataset[h][1])
        fr.append(dataset[h][2])
        rrt.append(dataset[h][3])
        hr.append(dataset[h][4])
        cld.append(dataset[h][5])
    for g in range(len(hr)):
        if (g == (len(hr)-1)):
            hrb.append(hr[g - 1])
            hra.append(0)
        elif (g == 0):
            hra.append(hr[g + 1])
            hrb.append(0)
        else:
            hra.append(hr[g + 1])
            hrb.append(hr[g - 1])
    for c in range(len(hr)-1):
        c+=1
        typenew = [amp[c], bk[c], fr[c], rrt[c], hrb[c], hr[c], hra[c], dtypes(cld[c])]
        dtn.append(typenew)
    return dtn
# "Amplitude","1 Back","1 Forward","RR","HR","HR interval",  "TYPE"
def writenmit(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\testMIT.csv', index=False)
def writentmit(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\fullMIT.csv', index=False)

def allinmit(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\allinMIT.csv', index=False)

def writen(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\test.csv', index=False)
def writent(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\full.csv', index=False)

def allin(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'E:\ECG\EKGReader\allin.csv', index=False)

def featuring(len_set,types):
    feature = []
    fc=[]
    for g in range(len(len_set)):
        print("Processing data " + str(g + 1))
        arv = (dataset_making(len_set[g],types))
        dir = r'E:\ECG\EKGReader'
        filed = '\\' + str(len_set[g])
        ftype = '.csv'
        filename = dir + filed + ftype
        pda = pd.DataFrame(arv, columns=["Amplitude", "1 Back", "1 Forward", "RR", "HR back", "HR", "HR After", "TYPE"])
        pda.to_csv(filename, index=False)
        for j in range(len(arv)-1):
            j+=1
            feature.append(arv[j])
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

def statistical_feature(feat):
    stat = []
    amp = []
    rr = []
    hr = []
    for w in range(len(feat)):
        amp.append(feat[w][0])
        rr.append(feat[w][3])
        hr.append(feat[w][5])
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
        stat.append([feat[k][0], feat[k][1] , feat[k][2], (amp[k]), feat[k][3], (rr[k]), feat[k][4], feat[k][5],feat[k][6], (hr[k]), feat[k][7]])
    return stat

def pecahdata(feat):
    result2=[]
    N = []
    V = []
    A = []
    F = []
    Q = []
    U = []
    for h in feat:
        if(h[5]==0):
            N.append(h)
        elif (h[5] == 1):
            V.append(h)
        elif (h[5] == 2):
            A.append(h)
        elif (h[5] == 3):
            F.append(h)
        elif (h[5] == 4):
            Q.append(h)
        else:
            U.append(h)
    result=[N,V,A,F,Q,U]
    for g in result:
        result2.extend(statistical_feature(g))
    return result2

def setunique(items):
    seen = set()
    for item in items:
        item = tuple(item)
        if item not in seen:
            yield item
            seen.add(item)
    return seen
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



def main():
    sg = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
          '116', '117', '118', '119', '121', '122', '123',
          '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217',
          '219', '220', '221', '222', '223', '228', '230',
          '231', '232', '233', '234']

    d1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118',
          '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223',
          '230']

    ds2 = ['100', '103', '105', '111', '113', '117', '121', '123',
           '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233'
        , '234']
    incart_all = ['I01','I02','I03','I04','I05','I06','I07','I08','I09','I10',
                  'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20',
                  'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30',
                  'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20',
                  'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40',
                  'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50',
                  'I51', 'I52', 'I53', 'I54', 'I55', 'I56', 'I57', 'I58', 'I59', 'I60',
                  'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70',
                  'I71', 'I72', 'I73', 'I74', 'I75'
                  ]
    incart_train = ['I01', 'I03',  'I05', 'I07',  'I09',
                  'I11',  'I13', 'I15', 'I17',  'I19',
                  'I21',  'I23', 'I25', 'I27',  'I29',
                  'I11', 'I13', 'I15', 'I17', 'I19',
                  'I31',  'I33', 'I35', 'I37', 'I39',
                  'I41',  'I43', 'I45', 'I47',  'I49',
                  'I51',  'I53', 'I55', 'I57',  'I59',
                  'I61',  'I63', 'I65', 'I67', 'I69',
                  'I71',  'I73',  'I75'
                  ]
    incart_test = ['I02',  'I04',  'I06',  'I08',  'I10',
                  'I12',  'I14', 'I16',  'I18',  'I20',
                  'I22',  'I24','I26',  'I28',  'I30',
                  'I12', 'I14',  'I16',  'I18',  'I20',
                  'I32',  'I34',  'I36',  'I38',  'I40',
                  'I42',  'I44',  'I46',  'I48',  'I50',
                  'I52', 'I54', 'I56',  'I58',  'I60',
                  'I62',  'I64',  'I66',  'I68',  'I70',
                  'I72', 'I74'
                  ]
    typ = input("1.Use MIT-BIH dataset\n2.Use INCART St.Petersburg Dataset\n3.Make all\nSelect data")
    if(typ=='1'):
        feature1 = featuring(d1,'1')
        writenmit(feature1)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature = featuring(ds2,'1')
        writentmit(feature)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature3 = featuring(sg,'1')
        # f3= statistical_feature(feature3)
        allinmit(feature3)
        exit()
    elif(typ==2):
        feature1 = featuring(incart_train,'2')
        writen(feature1)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature = featuring(incart_test,'2')
        writent(feature)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature3 = featuring(incart_all,'2')
        # f3= statistical_feature(feature3)
        allin(feature3)
        exit()
    else:
        feature1 = featuring(d1, '1')
        writenmit(feature1)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature = featuring(ds2, '1')
        writentmit(feature)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature3 = featuring(sg, '1')
        # f3= statistical_feature(feature3)
        allinmit(feature3)
        feature1 = featuring(incart_train, '2')
        writen(feature1)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature = featuring(incart_test, '2')
        writent(feature)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        feature3 = featuring(incart_all, '2')
        # f3= statistical_feature(feature3)
        allin(feature3)
        exit()



# for j in feature:
#     print (j)
# for j in ft:
#     print (j)
# f3=feature_weight(feature2)

# writenw(f3)
