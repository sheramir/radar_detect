"""Create results tables and generate output files for RF radar detection
algorithm.
Created on Sun May 31 2020
Version 1.0
@author: v025222357 Amir Sher
"""
import json
import numpy as np
import pandas as pd
from os import path

from utils import num_utils as nu
from utils import pandas_utils as pd_u
from utils import file_utils
from algo.algo_consts import *
from algo.RF_algo.get_radars_from_lib import get_radar_name


def generate_radar_results(result_signals_df, info_dict, select_radar_prob,
                           radar_lib, output_type, results_path=''):
    """Generate output files for radars detection algorithm.

    Parameters
    ----------
    result_signals_df : dataframe
        Dataframe of results table from parse_radars execution.
    info_dict : dictionary object
        Dictionary of execution info.
    file_name : string
        Xdat file name (for results info table).
    num_windows : integer
        Total number of windows that were calculated in the file (for results
        info table).
    def_win_size : integer
        Typical number of pulses per window (for results info table).
    select_radar_prob : float
        Required minimum occurances percentage of Radar in sub-signals (windows
    radar_lib : Dict
        Radar names library (from json file).
    output_type : string
        Required output file type ('Excel', 'CSV' or '').
    results_path : string, Optional.
        Results file name and path (without extension). Default is the name
        path of xdat.

    Returns
    -------
    None.

    """
    file_path = info_dict[INFO_FILE][:-5]
    if results_path == "":
        results_path = file_path
    else:
        results_path = path.join(results_path, file_utils.get_file_name(
            file_path))

    if (result_signals_df is None) and (output_type.upper() != 'BATCH'):
        # If no signals found just generate json file with basic file info
        results_dict = {INFO_TITLE: info_dict, JSON_SIGNALS_TITLE: {}}
        json_file = results_path + JSON_FILE_SUFFIX + '.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
    else:
        #Create pandas sum results tables
        column_names = [SIGNAL_TIME, SIGNAL_WAVEFORM, SIGNAL_RADAR_NAME,
                        SIGNAL_CENTRAL_FREQ, SIGNAL_OFFSET, SIGNAL_SNR,
                        SIGNAL_NUM_PULSES, SIGNAL_PRT, SIGNAL_PULSEWIDTH,
                        SIGNAL_BANDWIDTH, SIGNAL_LOW_FREQ, SIGNAL_HI_FREQ]
        signals_df = result_signals_df[column_names]

        #Create indexes and columns headers
        df_waveform_index = pd_u.get_unique_column_values(signals_df,
                                                          SIGNAL_WAVEFORM)
        cf_list = pd_u.get_unique_column_values(signals_df, SIGNAL_CENTRAL_FREQ)
        df_freq_columns = nu.format_num(cf_list)

        snr_header_list = [SNR_MEAN, SNR_MIN, SNR_MAX, SNR_STD]
        df_snr_freq_columns, df_snr_columns = pd_u.get_sub_headers(cf_list,
                                                                    snr_header_list)
        df_snr_freq_columns = nu.format_num(df_snr_freq_columns)

        #Create dataframes for output files
        total_win_df = pd_u.create_dataframe(index=df_waveform_index, columns=
                                             df_freq_columns)
        total_pulses_df = total_win_df.copy()
        total_snr_df = pd_u.create_dataframe(index=df_waveform_index,
                                             columns=(df_snr_freq_columns,
                                                      df_snr_columns))
        for indx in df_waveform_index:
            for freq in cf_list:
                col = nu.format_num(freq)
                filt = (signals_df[SIGNAL_CENTRAL_FREQ] == freq) & (signals_df[
                    SIGNAL_WAVEFORM] == indx)
                total_win_df.loc[indx, col] = sum(filt)
                total_pulses_df.loc[indx, col] = signals_df.num_pulses[filt].sum()
                snr = np.array(signals_df[SIGNAL_SNR][filt])

                if len(snr) > 0:
                    total_snr_df.loc[indx,
                                     (col, SNR_MEAN)] = round(snr.mean(), 2)
                    total_snr_df.loc[indx,
                                     (col, SNR_MIN)] = round(snr.min(), 2)
                    total_snr_df.loc[indx,
                                     (col, SNR_MAX)] = round(snr.max(), 2)
                    total_snr_df.loc[indx,
                                     (col, SNR_STD)] = round(snr.std(), 2)

        # remove empty columns from tables
        pd_u.remove_zero_columns(total_snr_df)
        pd_u.remove_zero_columns(total_win_df)
        pd_u.remove_zero_columns(total_pulses_df)

        # Generate Bottom line Results
        num_windows = info_dict[INFO_NUM_WINDOWS]
        results_dict = {INFO_TITLE: info_dict}
        total_df = total_win_df[(total_win_df / num_windows) > select_radar_prob]
        total_df_max = total_df.idxmax()

        radars_dict = {}
        i = 0
        for freq in df_freq_columns:
            if isinstance(total_df_max[freq], str):
                i += 1
                waveform = total_df_max[freq]
                radar_name = get_radar_name(waveform, radar_lib)
                occurrences = round(total_win_df[freq][waveform] / num_windows, 2)
                radar_dict = {JSON_SIGNAL_FREQUENCY: nu.format_num(freq),
                              JSON_SIGNAL_FREQUENCY_STR: freq,
                              JSON_RADAR_CLASSIFICATION: radar_name,
                              JSON_WAVEFORM: waveform,
                              JSON_RADAR_OCCURRENCES: occurrences,
                              JSON_RADAR_OCCURRENCES_STR: nu.format_num(
                                  occurrences, unit='')}
                radars_dict.update({i: radar_dict})
        results_dict.update({JSON_SIGNALS_TITLE: radars_dict})
        results_total_df = pd.DataFrame.from_dict(radars_dict)

        #Create info table
        info_df = pd.DataFrame.from_dict(info_dict, orient='index')

        #Generate Detailed Output Files
        if output_type.upper() == OUTPUT_EXCEL: #Create Excel file for data
            # analysis
            pd_u.df_to_excel(results_path,
                             (signals_df, total_win_df, total_pulses_df, total_snr_df,
                              results_total_df, info_df), EXCEL_FILE_SUFFIX,
                             (EXCEL_SHEET_FULLDATA, EXCEL_SHEET_WINDOWS,
                              EXCEL_SHEET_PULSES, EXCEL_SHEET_SNR,
                              EXCEL_SHEET_RESULTS,
                              EXCEL_SHEET_INFO))
        elif output_type.upper() == OUTPUT_CSV: # Create CSV files for data
            #analysis
            pd_u.df_to_csv(results_path,
                           (signals_df, total_win_df, total_pulses_df, total_snr_df),
                           (CSV_SHEET_FULLDATA, CSV_SHEET_WINDOWS, CSV_SHEET_PULSES,
                            CSV_SHEET_SNR))

        if output_type.upper() != OUTPUT_BATCH:
            #Create Json file with bottom line results
            json_file = results_path + JSON_FILE_SUFFIX + '.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=4)
        else:
            #Create file for batch execution
            sum_pulses_df = total_pulses_df.T.sum()
            unknowns = 0
            batch_index = int(info_dict[INFO_XDAT_NAME].split('_')[0])
            pulses = {BATCH_FIELD_RUNTIME: info_dict[INFO_ALGO_RUNTIME],
                      BATCH_FIELD_FILE: info_dict[INFO_XDAT_NAME],
                      BATCH_FIELD_KIPA: 0,
                      BATCH_FIELD_RADA: 0,
                      BATCH_FIELD_UNKNOWN: 0,
                      BATCH_FIELD_TOTAL: int(sum_pulses_df.sum()),
                      BATCH_FIELD_KIPA_SNR_MEAN: 0,
                      BATCH_FIELD_KIPA_SNR_STD: 0,
                      BATCH_FIELD_RADA_SNR_MEAN: 0,
                      BATCH_FIELD_RADA_SNR_STD: 0,
                      BATCH_FIELD_UNKNOWN_SNR_MEAN: 0,
                      BATCH_FIELD_UNKNOWN_SNR_STD: 0,
                      BATCH_FIELD_KIPA_SPREAD: 0,
                      BATCH_FIELD_RADA_SPREAD: 0,
                      BATCH_FIELD_UNKNOWN_SPREAD: 0
                      }
            snr_arr = {'Kipa': np.array([]),
                       'Rada': np.array([]),
                       'Unknown': np.array([])}
            for radar in sum_pulses_df.index:
                radar_name = get_radar_name(waveform, radar_lib, not_found=
                                            'Unknown')
                pulses[radar_name] = int(sum_pulses_df[radar])
                max_pulses = total_pulses_df.T.max()[radar]
                filt = (result_signals_df[SIGNAL_WAVEFORM] == radar)
                snr_arr[radar_name] = np.array([*snr_arr[radar_name], np.array(
                    result_signals_df.snr[filt])])
                if radar_name == 'Unknown':
                    unknowns = unknowns + total_pulses_df.T[radar] / max_pulses
                pulses[radar_name + '_spread'] = round(max_pulses / pulses[
                    radar_name], 2) if pulses[radar_name] > 0 else 0

            for radar, snr_data in snr_arr.items():
                pulses[radar + BATCH_FIELD_SNR_MEAN] = \
                    round(np.mean(snr_data), 2)
                pulses[radar + BATCH_FIELD_SNR_STD] = \
                    round(np.std(snr_data), 2)

            batch_df = pd.DataFrame.from_records([pulses], index=[batch_index],
                                                  columns=pulses.keys())
            csv_file = results_path
            if csv_file[-4:] != '.CSV':
                csv_file = csv_file + '.csv'

            if path.exists(csv_file):
                batch_df.to_csv(csv_file, mode='a', header=False)
            else:
                batch_df.to_csv(csv_file, mode='w')