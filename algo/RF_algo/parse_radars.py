"""Parse IQ file, search for radar signals and generate reports.
Created on Tue Apr 23 2020
Version 1.0
@author: v025222357 Amir Sher
"""
from algo.RF_algo.detect_radars_in_iq import detect_radars_in_iq
from algo.RF_algo.generate_radar_results import generate_radar_results
from algo.algo_consts import *


def parse_radars(xdat_file, iq_start=0, iq_length=0, output_type=OUTPUT_EXCEL,
                 select_radar_prob=0.6, default_pulses_in_window=15,
                 minimum_mf_score=50, correlation_width_factor=10, save_psd=False,
                 output_path=''):
    """Read xdat file and classify radars and waveforms in the file.

    Parameters
    ----------
    xdat_file : string
        Full name and path of input XDAT file.
    iq_start : float, optional
        Algorithm start time (sec) in the XDAT file. The default is 0.
    iq_length : float, optional
        Total length of signal (in secs) to execute algorithm. The default 0 =
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
    save_psd : boolean
        Saves PSD plots of each window to file. The default is False.
    output_path : string
        Folder path to hold result files

    Returns
    -------
    Boolean
        True if detected any signals, False if 0 signals detected.
    """
    signals, info, radars = detect_radars_in_iq(xdat_file, iq_start, iq_length,
                                               default_pulses_in_window,
                                               minimum_mf_score,
                                               correlation_width_factor,
                                               save_psd, output_path)

    generate_radar_results(signals, info, select_radar_prob, radars, output_type,
                           output_path)

    return signals is not None


############################
if __name__ == '__main__':
    import sys
    import getopt
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from timeit import default_timer as timer
    import os
    import glob
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

    if OUTPUT_TYPE == 'Batch': # execute on batch of files
        SAVE_PSD = False # Make sure no PSD is saved
        in_path = r'Z:\Testset\Noise\files' # or r'Z:\Testset\1K'
        batch_out = r'C:\Users\v025222357.TSGCO\Documents\Radar Detect\tests'
        csv_file = os.path.join(batch_out, 'radar_results_batch')

        num = -1
        xdat_files = glob.glob(os.path.join(in_path, '*.xdat'))
        num_files = len(xdat_files)

        for xdat_file in xdat_files:
            num += 1
            print(f'Parsing file {num} of {num_files}: {xdat_file}.')
            # Execute the main function
            start_timer = timer() # Calculate runtime
            signals, info, radars = detect_radars_in_iq(xdat_file, start, length,
                                                       DEFAULT_PULSES_IN_WINDOW,
                                                       MINIMUM_MF_SCORE,
                                                       CORRELATION_WIDTH_FACTOR,
                                                       SAVE_PSD)
            end_timer = timer() # Calculate detection runtime
            algo_runtime = round(end_timer - start_timer, 1)
            info.update({INFO_ALGO_RUNTIME: algo_runtime})

            if signals is not None:
                generate_radar_results(signals, info, SELECT_RADAR_PROB, radars,
                                       OUTPUT_TYPE, csv_file)
            gc.collect()
    else:
        #Parse command line parameters
        argv = sys.argv[1:]
        if argv == [] and input_file == "":
            print('parse_radars.py -i <xdat_file> -s <optional start time> -l <optional sample length>')
            sys.exit(2)
        try:
            opts, args = getopt.getopt(argv, 'hi:s:l:',
                                       ['help', 'file=', 'xdat=', 'start=',
                                        'length='])
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
        parse_radars(input_file, start, length, OUTPUT_TYPE,
                     SELECT_RADAR_PROB, DEFAULT_PULSES_IN_WINDOW,
                     MINIMUM_MF_SCORE, CORRELATION_WIDTH_FACTOR, SAVE_PSD)
        gc.collect()