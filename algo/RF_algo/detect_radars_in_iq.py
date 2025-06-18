"""Read IQ from xdat file, break IQ to small windows and identify radars in
windows.
Created on Tue Apr 23 2020
Version 1.0
@author: v025222357 Amir Sher
"""
from datetime import datetime
import numpy as np
import pandas as pd
import sys

from algo.RF_Classes.IQData import IQData as IQData
from algo.RF_algo.radar_detect import radar_detect
from utils import num_utils as nu
from algo.algo_consts import *
from algo.RF_algo.get_radars_from_lib import get_radars_waveforms_from_lib

# Configuration Constants
SCRIPT_VERSION = '2.0'
PLOT_FREQ = False
PLOT_NOISE = False
PLOT_MODULATE = False
PLOT_LOWPASS = False
PLOT_PULSES = False
PLOT_MATCHFILTER = False
ROLLING_WINDOW_FOR_NOISE = 500
ROLLING_WINDOW_FOR_MA = 300
MAXIMUM_DETECTION_DISTANCE = 500 # In Kilometer. Used to calculate
# max_possible_prt
C = 300000 # Speed of light km/s. Used to calculate max_possible_prt


def detect_radars_in_iq(xdat_file, iq_start=0, iq_length=0,
                        default_pulses_in_window=15, minimum_mf_score=50,
                        correlation_width_factor=10, save_psd=False, output_path=""):
    """Read xdat file and classify radars and waveforms in the file.

    Parameters
    ----------
    xdat_file : string
        Full name and path of input XDAT file.
    iq_start : float, optional
        Algorithm start time (sec) in the XDAT file. The default is 0.
    iq_length : float, optional
        Total length of signal (in secs) to execute algorithm. The default 0 ->
        until the end.
    output_type : string, optional
        Type of detailed results files ('Excel', 'CSV' or ''). The default is
        'Excel'.
    select_radar_prob : float
        Required minimum occurrences percentage of Radar in sub-signals
        (windows). The default is 0.6.
    default_pulses_in_window : TYPE, optional
        Typical number of pulses per window, to break down sub-signals. The
        default is 15.
    minimum_mf_score : float, optional
        Minimum matched filter cross-correlation score. The default is 50.
    correlation_width_factor : float, optional
        Minimum required matched filter correlation width compared to optimum.
        The default is 10.
    save_psd : Boolean
        Saves PSD plots of each window to file. The default is False.

    Returns
    -------
    signals_df : pandas dataframe
        Dataframe of signals found by the algorithm.
    info_dict : dictionary
        Information dictionary about the xdat file and algorithm execution.
    radar_lib : dictionary
        Dictionary of possible radars (from input json file).
    """
    if iq_length == 0:
        print_length = 'until end of file.'
    else:
        print_length = f'for {nu.format_num(iq_length)} seconds long.'
    print_begin = f'Reading xdat file from time {nu.format_num(iq_start)} sec '
    print(print_begin, print_length)

    full_IQ = IQData(xdat_file, iq_start, iq_length)
    if full_IQ.iq_path == "":
        print('Error getting xdat file')
        sys.exit(2)

    #full_IQ.plot_iq('time.real')
    data_end_time = full_IQ.t[-1]
    noise_full, _ = full_IQ.get_noise_level(w_len=ROLLING_WINDOW_FOR_NOISE)
    moving_average = full_IQ.moving_average(ROLLING_WINDOW_FOR_MA, 'abs')

    #Get waveforms and radar libraries from data files
    radar_lib, waveforms = get_radars_waveforms_from_lib()

    #Set default values
    num_pulses_def = default_pulses_in_window
    num_pulses_min = num_pulses_def - 3
    num_pulses_max = num_pulses_def + 3
    max_possible_prt = MAXIMUM_DETECTION_DISTANCE / C

    #Set initial values
    num_windows = 0
    win_start = 0.0
    win_length = 0.001
    prt = win_length / num_pulses_def
    rec_time = prt / 2
    min_window_l = 3 * prt
    min_window_r = 1 * prt
    num_pulses = 0
    check_begin_period = False
    last_round = False
    skip_round = False
    first_window = True
    signals_df = None

    while not last_round and win_start < data_end_time:
        #Search for start of pulses
        while num_pulses == 0 and not last_round:
            if (win_start + win_length) > data_end_time:
                win_length = data_end_time - win_start
                last_round = True
            if win_length < prt * num_pulses_min / 4:
                skip_round = True

            if not skip_round:
                sample_start = int(win_start * full_IQ.fs)
                sample_end = int((win_length + win_start) * full_IQ.fs)
                if max(moving_average[sample_start:sample_end]) > 2 * noise_full:
                    sub_IQ = full_IQ.copy(win_start, win_length)
                    widths, left_ips, right_ips, pulse_height, prt_pulse = \
                        sub_IQ.get_pulses(showplot=False)
                    num_pulses = len(widths)
                else: # only noise section - skip it
                    num_pulses = 0
            
            if num_pulses > 0:
                pw = np.percentile(widths, 50) # pulse width is median of all pulses
                prt = prt_pulse if prt_pulse > 0 else prt
                rec_time = prt - pw if prt_pulse > 0 else rec_time
                pulses_start = left_ips[0]
                pulses_end = right_ips[-1]
                begin_period = pulses_start - rec_time / 2
                end_period = win_length - pulses_end - rec_time / 2
                min_window_l = 3 * prt
                min_window_r = 1 * prt
            else:
                win_start = win_start + win_length
                win_length = num_pulses_def * prt
                check_begin_period = False

            if (not check_begin_period) and (not last_round):
                #Check if there are smaller pulses before the detected 1st pulse
                if begin_period >= min_window_l:
                    win_length = begin_period # shorten the window
                    num_pulses = 0 # trigger pulse search
                    check_begin_period = True
                # If the pulses end before the window end - shorten the window
                elif (end_period >= min_window_r) and (num_pulses > 1):
                    win_length = pulses_end + rec_time / 2
                    sub_IQ.cut_iq(0, win_length)
                #If too many pulses shorten the window and sub-signal
                elif num_pulses > num_pulses_max:
                    win_length = right_ips[num_pulses_def - 1] + rec_time / 2
                    sub_IQ.cut_iq(0, win_length)
                    num_pulses = num_pulses_def
                # If too few pulses try bigger window
                elif num_pulses < num_pulses_min:
                    win_length = 2 * win_length
                    if win_length < (max_possible_prt * num_pulses_def):
                        num_pulses = 0 # trigger new pulses search
                    else: # if win_length grows too long stop growing further
                        win_length = pulses_end + rec_time / 2
                        sub_IQ.cut_iq(0, win_length)
            
        if num_pulses > 0:
            num_windows += 1
            #Signal processing of current window:
            sub_IQ.start_time = win_start
            if output_path == '':
                output_path = full_IQ.iq_path

            signals = radar_detect(sub_IQ, waveforms, radar_lib,
                                   minimum_mf_score, correlation_width_factor,
                                   output_path, PLOT_FREQ, PLOT_NOISE,
                                   PLOT_MODULATE,
                                   PLOT_LOWPASS, PLOT_PULSES, PLOT_MATCHFILTER,
                                   save_psd)

            #Create dataframe of signals data
            num_signals = len(signals)
            if num_signals > 0:
                max_prt = max([v[SIGNAL_PRT] for k, v in signals.items()])

                if first_window:
                    signals_df = pd.DataFrame.from_dict(signals, orient='index')
                    signals_df.insert(0, SIGNAL_TIME, win_start)
                    first_window = False
                else:
                    df_sigl = pd.DataFrame.from_dict(signals, orient='index')
                    df_sigl.insert(0, SIGNAL_TIME, win_start)
                    signals_df = pd.concat([signals_df, df_sigl], ignore_index=
                                           True)
            else:
                max_prt = 0
            print(f'Time window {nu.format_num(win_start)}: Found {num_signals} signals. {num_pulses} pulses.')
            win_start = win_start + win_length
            prt = max_prt if max_prt > 0 else prt
            win_length = num_pulses_def * prt
            check_begin_period = False
            num_pulses = 0 # trigger pulse search
            del sub_IQ


    info_dict = {INFO_SCRIPT_VER: SCRIPT_VERSION,
                 INFO_FILE: full_IQ.iq_path,
                 INFO_FILE_PATH: full_IQ.file_path,
                 INFO_XDAT_NAME: full_IQ.xdat_name,
                 INFO_RUN_TIME: datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                 INFO_CAPTURE_START_TIME: str(full_IQ.start_capture),
                 INFO_RELATIVE_START_TIME: full_IQ.start_time,
                 INFO_ACQ_SCALE_FACTOR: full_IQ.acq_scale_factor,
                 INFO_SENSOR_CENTRAL_FREQUENCY: full_IQ.fc,
                 INFO_SAMPLING_FREQUENCY: full_IQ.fs,
                 INFO_SPAN: full_IQ.span,
                 INFO_NUMBER_OF_SAMPLES: full_IQ.num_samples,
                 INFO_IQ_LENGTH: round(data_end_time, 5),
                 INFO_NUM_WINDOWS: num_windows,
                 INFO_DEFAULT_WIN_SIZE: default_pulses_in_window}
    del full_IQ

    if signals_df is None:
        print(f'No pulses found in length {nu.format_num(data_end_time)} sec.')
    return signals_df, info_dict, radar_lib


#### ###
################ 
if __name__ == '__main__':
    #import sys
    import getopt
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import gc

    mpl.rcParams['figure.max_open_warning'] = 0 # allow many plots

    #Configuration Constants
    SELECT_RADAR_PROB = 0.6 # Minimum percentage of windows the radar is
    # detected (0-1)
    DEFAULT_PULSES_IN_WINDOW = 15 # Default required number of pulses in
    # sub-signal
    MINIMUM_MF_SCORE = 50 # Minimum Matched Filter correlation score (0-100)
    CORRELATION_WIDTH_FACTOR = 10 # Maximum Correlation width factor to pass
    # cross-correlation
    OUTPUT_TYPE = 'Excel' # possible values: 'Excel', 'CSV', 'Batch'
    SAVE_PSD = True # Save the PSD plots as PNG

    input_file = ""
    start = 0
    length = 0

    # Parse command line parameters
    argv = sys.argv[1:]
    if argv == [] and input_file == "":
        print('parse_radars.py -i <xdat_file> -s <optional start time> -l <optional sample length>')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(argv, 'hi:s:l:',
                                   ['help', 'file=', 'xdat=', 'start=', 'length='])
    except getopt.GetoptError:
        print('parse_radars.py -i <xdat_file> -s <optional start time> -l <optional sample length>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('parse_radars.py -i <xdat_file> -s <optional start time> -l <optional sample length>')
            sys.exit(0)
        elif opt in ('-i', '--file', '--xdat'):
            input_file = arg
        elif opt in ('-s', '--start'):
            start = float(arg)
        elif opt in ('-l', '--length'):
            length = float(arg)

    #Execute the main function
    detect_radars_in_iq(input_file, start, length, DEFAULT_PULSES_IN_WINDOW,
                        MINIMUM_MF_SCORE, CORRELATION_WIDTH_FACTOR, SAVE_PSD)
    gc.collect()