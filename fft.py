import wfdb
import pandas as pd
import numpy as np
import scipy.signal
def preprocessing(signal,samplerate):
    ecg = signal
    ransac_window_size = 5.0
    rate=samplerate
    # Low frequency of the band pass filter
    lowfreq = 5.0
    # High frequency of the band pass filter
    highfreq = 15.0
    lopass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    hipass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    # bandpass filtering
    ecg_lo = scipy.signal.filtfilt(*lopass, x=ecg)
    ecg_bd = scipy.signal.filtfilt(*hipass, x=ecg_lo)
    #cari turunan dan kuadratkan sinyal filter
    decg = np.diff(ecg_bd)
    decg_power = np.square(decg)
    return decg_power

def dataset_making(sig):
    signal = []
    dataset = []
    raw=[]
    ann = wfdb.rdann('INCART/' + sig, 'atr')
    sig, fields = wfdb.rdsamp('INCART/' + sig, channels=[5])
    for g in sig:
        raw.append(g[0])
    samplerate = ann.fs
    signew = preprocessing(raw,samplerate)
    cd = []
    for w in range(len(ann.sample)):
        types = ann.symbol[w]
        array = [ann.sample[w], types]
        cd.append(array)
    for k in range(len(signew)):
        sgt = signew[k]
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
def writen(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'..\EKGReader\test.csv', index=False)
def writent(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'..\EKGReader\full.csv', index=False)

def allin(set):
    pdr = pd.DataFrame(set, columns=["Amplitude","1 Back","1 Forward","RR","HR back","HR","HR After", "TYPE"])
    pdr.to_csv(r'..\EKGReader\allin.csv', index=False)

def featuring(len_set):
    feature = []
    fc=[]
    for g in range(len(len_set)):
        print("Processing data " + str(g + 1))
        arv = (dataset_making(len_set[g]))
        dir = r'..\EKGReader'
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
    feature1 = featuring(incart_train)
    writen(feature1)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    feature = featuring(incart_test)
    writent(feature)
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    feature3 = featuring(incart_all)
    allin(feature3)
    exit()




