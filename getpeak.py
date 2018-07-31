import pandas as pd
import numpy as np
from wfdb import processing as ps
import json

def load_tester(path):
    with open(path) as f:
        data = json.load(f)
    print(data)
    return np.asarray(data)



def makefeat(signal):
    peakers = ps.xqrs_detect(signal, 100)
    real_peak = peakers
    preprocessed = []
    for k in real_peak:
        preprocessed.append(signal[k])
    samplerate = 100
    rr = []
    HR = []
    for w in range(len(real_peak)):
        if (w < len(real_peak) - 1):
            rrs = (real_peak[w + 1] - real_peak[w]) / samplerate
            HRs = 60 / rrs
            rr.append(rrs)
            HR.append(HRs)
        else:
            rr.append(0)
            HR.append(0)

    threeclash = []
    for j in range(len(rr)):
        peak = preprocessed[j]
        if(j==0):
            toback=0
            toward = preprocessed[j]-preprocessed[j+1]
        elif(j==len(rr)-1):
            toback = preprocessed[j] - preprocessed[j - 1]
            toward = 0
        rint = rr[j]
        Hrt = HR[j]
        temp = [peak,toback,toward,rint, Hrt]
        threeclash.append(temp)

    featureist = []
    va=0
    vb=0
    amp = 0
    hrv = 0
    rrv = 0
    for k in threeclash:
        amp += k[0]
        rrv += k[3]
        hrv += k[4]
    ampv = amp / len(threeclash)
    rrvv = rrv / len(threeclash)
    hrvv = hrv / len(threeclash)
    vg = len(threeclash)
    scr = float(vg/15)
    for k in range (len(threeclash)):
        if(k==0):
            va = threeclash[k + 1][0]
            vb = 0
            hb = 0
            hf = threeclash[k+1][4]

        elif(k==len(threeclash)-1):
            hb = threeclash[k][3]-threeclash[k - 1][4]
            hf = 0
            va = 0
            vb = threeclash[k - 1][0]

        else:
            hb = threeclash[k][3]-threeclash[k - 1][4]
            hf = threeclash[k][3]-threeclash[k+1][4]
            va = threeclash[k + 1][0]
            vb = threeclash[k - 1][0]

        # avar = np.absolute(threeclash[k][0] - ampv)
        # rvar = np.absolute(threeclash[k][1] - rrvv)
        # hrvar = np.absolute(threeclash[k][2] - hrvv)

        temp = [threeclash[k][0],vb,va,threeclash[k][1],hb,threeclash[k][3],hf]
        featureist.append(temp)
    return featureist

def signalmaker():
    fileset = ["samp1","samp2","samp3","samp4","samp5"]
    for w in fileset:
        filename = "\\" + str(w)
        dir = r"E:\ECG\EKGReader\Data\json"
        type = ".json"
        path = dir + filename + type
        typeout = ".csv"
        dirs = dir + filename + typeout
        signal = load_tester(path)
        featureist = makefeat(signal)
        pdr = pd.DataFrame(featureist, columns=["Amplitude","1 Back","1 Forward","RR","HR","HR Before","HR After"])
        pdr.to_csv(dirs, index=False)
    exit()