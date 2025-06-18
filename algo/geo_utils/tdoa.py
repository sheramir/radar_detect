"""Calculate time delay between signals.
Created on Wed Jun 24 16:50:43 2020
@author: v025222357 Amir Sher
"""
import numpy as np
from scipy import signal
from scipy.constants import c # speed of light
import matplotlib.pyplot as plt


def tdoa(x, y, ts, showplot=False):
    """Calculate delay (Time Difference of Arrival) between 2 synchronized IQ
    arrays.

    Parameters
    ----------
    x : numpy array
        IQ data from sensor X.
    y : numpy array
        IQ data from sensor Y.
    ts : float
        Sampling time of IQ files (1/fs).
    showplot : boolean, optional
        Show plot of the correlation between IQ arrays. The default is False.

    Returns
    -------
    delay : float
        Time delay between x and y (seconds).
    distance : float
        Calculated distance difference between the sensors.
    """
    if np.iscomplex(x).any():
        x = abs(x)
    if np.iscomplex(y).any():
        y = abs(y)

    x = x - np.mean(x)
    y = y - np.mean(y)

    corr = signal.correlate(x, y, mode='full')
    max_corr = np.argmax(corr)

    delay = (len(x) - max_corr) * ts
    distance = c * delay

    if showplot:
        fig, ax = plt.subplots()
        ax.set_title(f'TDOA Signals Correlation')
        t_axis = np.arange(len(x) - 1, -len(x), -1) * ts
        ax.plot(t_axis, corr / max(corr), 'b')
        ax.grid(True)
        plt.show()

    return delay, distance