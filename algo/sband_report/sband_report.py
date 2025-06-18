"""Generate S-Band report dataframes from json files created by parse_radars
algorithm.
Created on Mon Jul 13 17:42:28 2020
@author: v025222357 Amir Sher
"""
import pandas as pd
from algo.algo_consts import *
from consts import *
from algo.sband_report.decision_tree import get_decision
from algo.sband_report.sband_plot import sband_plot

#Constants
SBAND_MIN = 3.1e9
SBAND_MAX = 3.5e9


def cf_main_or_secondary(freq, cf, span):
    """Check if frequency is considered in main capture or secondary
    (overlapping) capture compared to central frequency.

    Parameters
    ----------
    freq : float
        Frequency to validate (in Hz).
    cf : float
        Central frequency of the sensor (in Hz).
    span : float
        Span of the sensor capture (in Hz).

    Returns
    -------
    String
        Position of frequency 'main', 'second' (or 'none' if out of band).
    """
    diff = abs(freq - cf)
    if diff <= (span / 4):
        return SBAND_MAIN
    elif diff <= (span / 2):
        return SBAND_SECOND
    else:
        return 'none'


def json_to_df(jsonfile):
    """Parse a json data file (output of parse_radars algorithm).

    Parameters
    ----------
    jsonfile : dict
        Data created from running parse_radars algorithm on one IQ file.

    Returns
    -------
    df_main : dataframe
        dataframe containing frequecies and waveforms from the json that are
        Main to the central frequency.
    df_second : dataframe
        dataframe containing frequecies and waveforms from the json that are
        Second to the central frequency.
    df_info : dataframe
        Information about the capture (time, sensor, span, central_frequency).
    """
    info = jsonfile[INFO_TITLE]
    sensor_cf = info[INFO_SENSOR_CENTRAL_FREQUENCY]
    span = info[INFO_SPAN]
    capture_time = info[INFO_CAPTURE_START_TIME]

    try:
        sensor_name = info[SENSOR_NAME_HKUBE_ELASTIC]
    except KeyError:
        sensor_name = SBAND_UNDEFINED_SENSOR

    data = {SBAND_MAIN: {SBAND_FREQUENCY: [], SBAND_CLASS_MAIN: [],
                         SBAND_SCORE_MAIN: []},
            SBAND_SECOND: {SBAND_FREQUENCY: [], SBAND_CLASS_SECOND: [],
                           SBAND_SCORE_SECOND: []},
            INFO_TITLE: {INFO_SENSOR_CENTRAL_FREQUENCY: [sensor_cf],
                         SENSOR_NAME_HKUBE_ELASTIC: [sensor_name], INFO_SPAN: [span],
                         INFO_CAPTURE_START_TIME: [capture_time]}}
    try:
        signals = jsonfile[JSON_SIGNALS_TITLE]
        for i, signal in signals.items():
            sig_freq = signal[JSON_SIGNAL_FREQUENCY]
            sig_wave = signal[JSON_WAVEFORM]
            sig_score = 100 * signal[JSON_RADAR_OCCURRENCES]
            sig_pos = cf_main_or_secondary(sig_freq, sensor_cf, span)
            if sig_pos != 'none':
                data[sig_pos][SBAND_FREQUENCY].append(sig_freq)
                data[sig_pos]['class' + sig_pos].append(sig_wave)
                data[sig_pos]['score' + sig_pos].append(sig_score)
                data[INFO_TITLE].update({SBAND_SIGNAL + str(i): [sig_wave]})
    except KeyError:
        pass

    df_main = pd.DataFrame.from_records(data[SBAND_MAIN], index=SBAND_FREQUENCY)
    df_second = pd.DataFrame.from_records(data[SBAND_SECOND], index=
                                          SBAND_FREQUENCY)
    df_info = pd.DataFrame.from_records(data[INFO_TITLE], index=
                                        INFO_SENSOR_CENTRAL_FREQUENCY)
    return df_main, df_second, df_info


def sband_report(jsons_list, filepath, saveplot=True):
    """Build integrated dataframe from all json files generated with
    parse_radars algorithm on different central frequencies.

    Parameters
    ----------
    jsons_list : list
        list of jsons (dicts) generated with parse_radars algorithm.
    filepath : string
        The target path for result files.

    Returns
    -------
    df_result, df_info : Results dataframes.
    """
    data = {SBAND_MAIN: {SBAND_FREQUENCY: [], SBAND_CLASS_MAIN: [],
                         SBAND_SCORE_MAIN: []},
            SBAND_SECOND: {SBAND_FREQUENCY: [], SBAND_CLASS_SECOND: [],
                           SBAND_SCORE_SECOND: []},
            INFO_TITLE: {INFO_SENSOR_CENTRAL_FREQUENCY: [], INFO_SPAN: [],
                         INFO_CAPTURE_START_TIME: []}}
    df_main = pd.DataFrame.from_records(data[SBAND_MAIN], index=SBAND_FREQUENCY)
    df_second = pd.DataFrame.from_records(data[SBAND_SECOND], index=
                                          SBAND_FREQUENCY)
    df_info = pd.DataFrame.from_records(data[INFO_TITLE], index=
                                        INFO_SENSOR_CENTRAL_FREQUENCY)

    for jsonfile in jsons_list:
        df_main_json, df_second_json, df_info_json = json_to_df(jsonfile)
        df_main = pd.concat([df_main, df_main_json])
        df_second = pd.concat([df_second, df_second_json])
        df_info = pd.concat([df_info, df_info_json])

    df_sband = pd.concat([df_main, df_second], axis=1)
    nan_values = {SBAND_CLASS_MAIN: SBAND_SIGNAL_NAN, SBAND_SCORE_MAIN: 0,
                  SBAND_CLASS_SECOND: SBAND_SIGNAL_NAN, SBAND_SCORE_SECOND: 0}
    df_sband = df_sband.fillna(value=nan_values)

    sensor = set(df_info[SENSOR_NAME_HKUBE_ELASTIC])
    span = set(df_info[INFO_SPAN])

    if len(sensor) > 1:
        print('ERROR: Input from different sensors!')
        return False
    if len(span) > 1:
        print('ERROR: Recordings with different Span!')
        return False
    else:
        span = max(span)

    central_frequencies = list(range(int(SBAND_MIN), int(SBAND_MAX), int(span
                                                                          / 2)))
    df_cf = pd.DataFrame.from_records({INFO_SENSOR_CENTRAL_FREQUENCY:
                                       central_frequencies}, index=INFO_SENSOR_CENTRAL_FREQUENCY)
    df_info = pd.merge(df_cf, df_info, how='left', on=
                       INFO_SENSOR_CENTRAL_FREQUENCY)

    count_nulls = len(df_info[df_info[INFO_SPAN].isnull()])
    count_rows = len(df_info)

    if count_nulls < count_rows:
        df_result = df_sband.apply(get_decision, axis=1, result_type='expand')
        df_result.columns = [SBAND_CLASS_FINAL, SBAND_SCORE_FINAL]
        #Generate plot and save CSV files:
        sband_plot(df_result, df_info, filepath, saveplot)
        return df_result, df_info
    else:
        print('Error: data is empty!')
        return False


######## ######
########################
if __name__ == '__main__':
    from os import path
    from get_jsons import get_jsons

    jsons_list = get_jsons()
    filepath = path.dirname(path.abspath(__file__))
    df_result, df_info = sband_report(jsons_list, filepath, saveplot=True)

    #Raw data from jsons:
    # frequency   class_main  score_main class_second  score_second
    #3.140000e+09    LFM-down        71.0       LFM-up          81.0
    #3.210000e+09  Barker13        95.0     Barker13          86.0
    #3.290000e+09    Unknown        61.0      Unknown          91.0
    #3.340000e+09      LFM-up        71.0       LFM-up          91.0
    #3.360000e+09  Barker13       100.0     Barker13          81.0
    #3.420000e+09  Barker13        81.0     Barker13          86.0
    #3.440000e+09    LFM-down        91.0            X           0.0

    #Results of final get_decision apply:
    # frequency  signal_class  signal_score
    #3.140000e+09       LFM-up         63.25
    #3.210000e+09     Barker13        100.00
    #3.290000e+09      Unknown         76.00
    #3.340000e+09        LFM-up        100.00
    #3.360000e+09      Barker13        100.00
    #3.420000e+09      Barker13        100.00
    #3.440000e+09      LFM-down         91.00