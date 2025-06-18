"""Detect and identify radars in small window (from a partial IQ file).
Created on Tue Apr 23 2020
Version: 1.0
@author: v025222357 Amir Sher
"""
import numpy as np
from algo.RF_Classes.IQData import IQData as IQData
from algo.RF_Classes import matched_filter as mf
from algo.algo_consts import *
from algo.RF_algo.get_radars_from_lib import get_radar_name


def radar_detect(myIQ, waveforms, radar_lib,
                 minimum_mf_score=50, xcorr_width_factor=10, file_path='',
                 plot_freq=False, plot_noise=False, plot_modulate=False,
                 plot_lowpass=False, plot_pulses=False, plot_matchfilter=False,
                 save_psd=False):
    """Detect and identify radar pulses in sub-IQ window of small size.

    Parameters
    ----------
    myIQ : IQData class
        The IQ to test.
    waveforms : dict
        dictionary of possible radar waveforms for matched filtering (from json
        file).
    radar_lib : dict
        dictionary of radar name and their associated waveform.
    minimum_mf_score : float, optional
        Minimum matched filter cross-correlation score. The default is 50.
    xcorr_width_factor : float, optional
        Minimum required matched filter correlation width compared to optimum.
        The default is 10.
    plot_freq : boolean, optional
        Plot the PSD plot. The default is False.
    plot_noise : boolean, optional
        Plot the noise plot. The default is False.
    plot_modulate : boolean, optional
        Plot the modulation plot. The default is False.
    plot_lowpass : boolean, optional
        Plot the lowpass filter plot. The default is False.
    plot_pulses : boolean, optional
        Plot the time pulses plot. The default is False.
    plot_matchfilter : boolean, optional
        Plot the matched filter plot. The default is False.
    save_psd : boolean, optional
        Save the PSD plot to PNG file. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    myIQ.get_freq_signals(showplot=plot_freq)
    if save_psd:
        myIQ.plot_iq(save=True, path=file_path, plot1='psd')

    noise, _ = myIQ.get_noise_level(showplot=plot_noise)

    for sig_i in myIQ.signals:
        mysignal = myIQ.signals[sig_i]
        sig_offset = mysignal[SIGNAL_OFFSET]
        sig_bw = mysignal[SIGNAL_BANDWIDTH]
        sig_flo = mysignal[SIGNAL_LOW_FREQ]

        one_sig = myIQ.copy()
        one_sig.new_iq(one_sig.modulate(-sig_offset, showplot=plot_modulate))
        one_sig.new_iq(one_sig.butter_low_pass(sig_bw / 2, order=10, margin=20,
                                               showplot=plot_lowpass))

        _, widths, _, pulse_height, prt = one_sig.get_pulses(showplot=plot_pulses)
        num_pulses = len(widths)
        pw = round(np.median(widths), 7) if num_pulses > 0 else 0 # pulse width
        # is median of all pulses
        p_rms = np.median(pulse_height) if num_pulses > 0 else 0 # median of
        # pulse height RMS
        snr = round(20 * np.log10(p_rms / noise), 1) if num_pulses > 0 else 0

        myIQ.signals[sig_i].update({SIGNAL_NUM_PULSES: num_pulses})
        myIQ.signals[sig_i].update({SIGNAL_PULSEWIDTH: pw})
        myIQ.signals[sig_i].update({SIGNAL_PRT: prt})
        myIQ.signals[sig_i].update({SIGNAL_SNR: snr})

        fs = one_sig.fs
        max_score = 0
        mf_name = RADAR_UNKNOWN
        radar_name = RADAR_UNKNOWN

        if len(widths) > 0: #Search waveform library for matched filter
            for wav_i in waveforms: # Loop and test signal against all MP waveforms
                mod_type = waveforms[wav_i]['mod_type']
                sweep_dir = waveforms[wav_i]['sweep_dir']
                chips = waveforms[wav_i]['chips']
                test_mf = mf.MatchedFilter(mod_type=mod_type, fs=fs, offset=0,
                                           pw=pw, bw=sig_bw,
                                           chips=chips, sweep_dir=sweep_dir)

                corr, corr_pass, scores, _ = test_mf.xcorr(one_sig.iq,
                                                           normalize=p_rms,
                                                           score=minimum_mf_score,
                                                           min_width_factor=
                                                           xcorr_width_factor,
                                                           showplot=plot_matchfilter)

                if corr_pass:
                    new_score = max(scores)
                    if new_score > max_score:
                        max_score = new_score
                        mf_name = test_mf.mf_name

                del test_mf

        if max_score > 0: #At least one waveform matched the signal
            #Get name of radar from library according to its waveform type
            # If we can identify the waveform but it's not in the radar name
            # library, then the radar name is Unlisted
            radar_name = get_radar_name(mf_name, radar_lib)
        else:
            # The signal was not matched by any of the tested waveforms
            radar_name = RADAR_UNIDENTIFI

        myIQ.signals[sig_i].update({SIGNAL_WAVEFORM: mf_name})
        myIQ.signals[sig_i].update({SIGNAL_RADAR_NAME: radar_name})

        del one_sig

    return myIQ.signals


#########
def main():
    import json
    import matplotlib as mpl

    mpl.rcParams['figure.max_open_warning'] = 0

    with open('radarlib.json', 'r', encoding='utf-8') as f:
        radar_lib = json.load(f)

    with open('waveformlib.json', 'r', encoding='utf-8') as f:
        waveforms = json.load(f)

    start_time = 0.00015
    sig_length = 0.002625
    repository = 'C:\\IQ_Data\\Radar\\'
    sensor = 'Kisufim\\'
    radar = '' # Assuming this is meant to be empty or a variable
    sensor_file = '21_RADA+KIPA2020-01-01_23-35-25.xdat' # 560MB about 2.3 sec
    xdat_file = repository + sensor + radar + sensor_file
    #xdat_file = 'C:\IQ_Data\Radar\Kisufim\RADA+KIPA2020-01-01_20-58-35B.xdat'

    print('Display only on RadarDetect exec')
    IQfile = IQData(xdat_file, start_time, sig_length)
    IQfile.pwelch(showplot=True)
    IQfile.plot_iq(plot1='time.real')

    signals = radar_detect(IQfile, waveforms, radar_lib)

    print(signals)


if __name__ == '__main__':
    main()