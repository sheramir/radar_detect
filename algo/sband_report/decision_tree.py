"""Decide between two signals data captured with overlapping frequency range.
Created on Tue Jul 14 12:01:11 2020
@author: v025222357 Amir Sher
"""
import pandas as pd
from os import path
from algo.algo_consts import *
import algo.RF_algo.get_radars_from_lib as get_radars
from utils import list_utils

decision_tree_file = path.join(path.dirname(path.abspath(__file__)),
                               'radar_decision_table.csv')


def build_decision_dataframe(decision_table=decision_tree_file):
    """Build decision table with all combinations of waveforms. The logic is
    coded in a CSV file.

    Parameters
    ----------
    decision_table : string, optional
        Path of the CSV file that includes the decision logic.

    Returns
    -------
    df_full : dataframe
        A detailed dataframe for decision tree with all waveforms combinations.
    """
    df_source = pd.read_csv(decision_table)
    nan_values = {SBAND_CLASS_MAIN: SBAND_SIGNAL_NAN,
                  SBAND_CLASS_SECOND: SBAND_SIGNAL_NAN,
                  SBAND_CLASS_FINAL: SBAND_SIGNAL_NAN}
    df_source = df_source.fillna(value=nan_values)

    radar_names, radar_waveforms = get_radars.get_radars_list()
    all_waveforms = get_radars.get_waveforms_list()
    unlisted_waveforms = list_utils.remove_list_from_list(all_waveforms,
                                                          radar_waveforms)

    all_possibilities = [*all_waveforms, SBAND_SIGNAL_UNKNOWN, SBAND_SIGNAL_NAN]
    permutations = list_utils.permutate(all_possibilities)
    class_main_list, class_second_list = list_utils.pairs_to_lists(permutations)

    new_dict = {SBAND_CLASS_MAIN: class_main_list,
                SBAND_CLASS_SECOND: class_second_list,
                SBAND_CLASS_FINAL: [],
                CSV_DELTA: [],
                SBAND_SCORE_FINAL: []}

    for perm in permutations:
        class_main_code = perm[0]
        class_second_code = perm[1]

        if perm[0] in radar_waveforms:
            class_main_code = CSV_LISTED_WAVE
        elif perm[0] in unlisted_waveforms:
            class_main_code = CSV_UNLISTED_WAVE
        elif perm[0] == SBAND_SIGNAL_UNKNOWN:
            class_main_code = CSV_UNKNOWN

        if (perm[1] in radar_waveforms) and (perm[0] in radar_waveforms) and (
                perm[1] != perm[0]):
            class_second_code = CSV_LISTED_WAVE_SECOND
        elif perm[1] in radar_waveforms:
            class_second_code = CSV_LISTED_WAVE
        elif (perm[1] in unlisted_waveforms) and (perm[0] in unlisted_waveforms) \
                and (perm[1] != perm[0]):
            class_second_code = CSV_UNLISTED_WAVE_SECOND
        elif perm[1] in unlisted_waveforms:
            class_second_code = CSV_UNLISTED_WAVE
        elif perm[1] == SBAND_SIGNAL_UNKNOWN:
            class_second_code = CSV_UNKNOWN

        filt = (df_source[SBAND_CLASS_MAIN] == class_main_code) & \
               (df_source[SBAND_CLASS_SECOND] == class_second_code)
        score_str = df_source[filt][SBAND_SCORE_FINAL].sum()
        class_final_str = df_source[filt][SBAND_CLASS_FINAL].sum()
        delta = df_source[filt][CSV_DELTA].sum()

        if class_final_str == class_main_code:
            class_final_str = perm[0]
        elif class_final_str == class_second_code:
            class_final_str = perm[1]

        new_dict[SBAND_CLASS_FINAL].append(class_final_str)
        new_dict[CSV_DELTA].append(delta)
        new_dict[SBAND_SCORE_FINAL].append(score_str)

    df_full = pd.DataFrame.from_records(new_dict)
    return df_full


df_full = build_decision_dataframe()


def get_decision(row, df=df_full):
    """Return the chosen signal and score, based on the logic in the input df.
    This function is used with apply method of a dataframe of signals.

    Parameters
    ----------
    row : TYPE
        Row of the input dataframe.
    df : TYPE, optional
        The decision table dataframe that is built by build_decision_dataframe.
        The default is df_full.

    Returns
    -------
    class_str : string
        The chosen waveform type.
    score : float
        The score of the chosen waveform.
    """
    filt = (df[SBAND_CLASS_MAIN] == row[SBAND_CLASS_MAIN]) & \
           (df[SBAND_CLASS_SECOND] == row[SBAND_CLASS_SECOND])
    score_str = df[filt][SBAND_SCORE_FINAL].sum()
    class_str = df[filt][SBAND_CLASS_FINAL].sum()
    delta = df[filt][CSV_DELTA].sum()
    main_score = row[SBAND_SCORE_MAIN]
    second_score = row[SBAND_SCORE_SECOND]

    if class_str == 'max':
        if (main_score + delta) > second_score:
            class_str = row[SBAND_CLASS_MAIN]
            score = eval(score_str.replace('main_score', str(main_score)).replace('second_score', str(second_score)))
        else:
            class_str = row[SBAND_CLASS_SECOND]
            score = eval(score_str.replace('main_score', str(main_score)).replace('second_score', str(second_score)))
    else:
        score = eval(score_str.replace('main_score', str(main_score)).replace('second_score', str(second_score)))
    return class_str, score


if __name__ == '__main__':
    pass