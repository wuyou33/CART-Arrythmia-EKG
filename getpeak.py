import numpy as np
from wfdb import processing as ps
import json
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage

def load_tester(path):
    with open(path) as f:
        data = json.load(f)
    data=np.asarray(data)
    return data.astype(np.float32)


def detect_beats(signal):
    ecg = []
    for g in range(5000):
        ecg.append(signal[g])
    ransac_window_size = 5.0
    rate=100
    # Low frequency of the band pass filter
    lowfreq = 5
    # High frequency of the band pass filter
    highfreq = 15
    #random sampling cosensus windows size
    ransac_window_size = int(ransac_window_size * rate)

    lopass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
    hipass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')
    # TODO: Band pass filter disini
    ecg_lo = scipy.signal.filtfilt(*lopass, x=ecg)
    ecg_bd = scipy.signal.filtfilt(*hipass, x=ecg_lo)

    # Square (=signal power) turunan sinyal, dan dikuadratkan
    decg = np.diff(ecg_bd)
    decg_power = decg ** 2

    # Thresholding dan normalisasi sinyal
    thresholds = []
    max_powers = []
    for i in range(int(len(decg_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = decg_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))

    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    decg_power[decg_power < threshold] = 0

    decg_power /= max_power
    decg_power[decg_power > 1.0] = 1.0
    square_decg_power = decg_power ** 2

    #kalkulasi energi Shannon
    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0

    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
    # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)

    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 8.0)
    lp_energy_diff = np.diff(lp_energy)

    #posisi zero crossing sinyal, posisi peak
    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    zero_crossings -= 1
    return zero_crossings


def makefeat(signal,result):
    real_peak =  result
    samplerate = 100
    rr = []
    HR = []
    signals=[]
    for k in result:
        signals.append(signal[k])
    for w in range(len(real_peak)):
        if (w < len(real_peak) - 1):
            rrs = round(((real_peak[w + 1] - real_peak[w]) / samplerate),3)
            HRs = int(60 / rrs)
            rr.append(rrs)
            HR.append(HRs)
        else:
            rr.append(0)
            HR.append(0)
    threeclash = []
    for j in range(len(rr)):
        peak = signals[j]
        if(j==0):
            toback = 0
            toward = signals[j+1]
        elif(j==len(rr)-1):
            toback = signals[j - 1]
            toward = 0
        else:
            toward = signals[j + 1]
            toback = signals[j - 1]
        rint = rr[j]
        Hrt = HR[j]
        temp = [peak,toback,toward,rint, Hrt]
        threeclash.append(temp)

    featureist = []
    for k in range (len(threeclash)):
        if(k==0):
            hb = 0
            hf = threeclash[k+1][4]

        elif(k==len(threeclash)-1):
            hb = threeclash[k - 1][4]
            hf = 0
        else:
            hb = threeclash[k - 1][4]
            hf = threeclash[k+1][4]
        temp = [threeclash[k][0],threeclash[k][1],threeclash[k][2],threeclash[k][3],hb,threeclash[k][4],hf]
        featureist.append(temp)
    return featureist


