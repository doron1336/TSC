import obspy
from obspy.signal.trigger import plot_trigger, recursive_sta_lta
import numpy as np
import pandas as pd


st = obspy.read('./201804231257_BTSH.mseed')  # load example seismogram
# tr = st[0]
# print(tr)
# print(st)
# st.filter(type='highpass', freq=3.0)
# st = st.select(component='Z')
# st.plot()

tr = st.select(component="Z")[0]
# tr.filter("bandpass", freqmin=1, freqmax=20)  

# tr.plot()
# print(len(tr))
# print(tr.stats)

df = pd.DataFrame(tr.data[800:1200])
print(df.head())

def classic_sta_lta_py(a, nsta, nlta):
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.
    .. note::
        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!
    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2, dtype=np.float64)
    # print(nsta)
    # print(nlta)
    # print(sta)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    g = sta[nsta:]
    h = sta[:-nsta] 
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


sta = 0.5
lta = 4

# print(tr.data.shape)
cft = classic_sta_lta_py(tr.data, int(sta * tr.stats.sampling_rate), int(lta * tr.stats.sampling_rate))

# thrOn = 4
# thrOff = 0.7
# plot_trigger(tr, cft, thrOn, thrOff)

