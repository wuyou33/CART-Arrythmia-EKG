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
        rint = rr[j]
        Hrt = HR[j]
        temp = [peak, rint, Hrt]
        threeclash.append(temp)

    featureist = []
    amp = 0
    hrv = 0
    rrv = 0
    for k in threeclash:
        amp += k[0]
        rrv += k[1]
        hrv += k[2]
    ampv = amp / len(threeclash)
    rrvv = rrv / len(threeclash)
    hrvv = hrv / len(threeclash)

    for k in threeclash:
        avar = np.absolute(k[0] - ampv)
        rvar = np.absolute(k[1] - rrvv)
        hrvar = np.absolute(k[2] - hrvv)
        temp = [k[0], avar, k[1], rvar, k[2], hrvar]
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
        pdr = pd.DataFrame(featureist, columns=["Amplitude", "Var Amp", "RR", "Var RR", "HR", "Var HR"])
        pdr.to_csv(dirs, index=False)
    exit()