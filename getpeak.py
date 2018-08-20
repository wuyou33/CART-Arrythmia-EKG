import numpy as np
from wfdb import processing as ps
import json
import matplotlib.pyplot as plt

def load_tester(path):
    with open(path) as f:
        data = json.load(f)
    data=np.asarray(data)
    return(data.astype(np.float32))


def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = ps.compute_hr(sig_len=len(sig), qrs_inds=peak_inds, fs=fs)

    N = sig.shape[0]

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    ax_left.plot(sig, color='#3979f0', label='Signal')
    ax_left.plot(peak_inds, sig[peak_inds], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    ax_right.plot(np.arange(N), hrs, label='Heart rate', color='m', linewidth=2)

    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
    ax_right.set_ylabel('Heart rate (bpm)', color='m')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    ax_right.tick_params('y', colors='m')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()

def makefeat2(signal):
    peakers = ps.xqrs_detect(signal, 100)
    return peakers

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


