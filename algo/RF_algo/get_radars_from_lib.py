"""Functions that return info about known radars and known waveforms from
json files.

Created on Wed Jul 15 12:02:21 2020
@author: v025222357 Amir Sher
"""
import json
from os import path
from algo.algo_consts import *


def get_radars():
    """Get listed radar names and their waveforms from json files.

    Returns
    -------
    radar_lib : dict
        Dictionary of Radars from radarlib.json.
    """
    current_dir = path.dirname(__file__) # Current module directory
    with open(path.join(current_dir, 'radarlib.json'), 'r', encoding='utf-8') \
            as f:
        radar_lib = json.load(f)
    return radar_lib


def get_waveforms():
    """Get listed waveforms from json files.

    Returns
    -------
    waveforms : dict
        Dictionary of waveforms from waveformlib.json.
    """
    current_dir = path.dirname(__file__) # Current module directory
    with open(path.join(current_dir, 'waveformlib.json'), 'r', encoding='utf-8') \
            as f:
        waveforms = json.load(f)
    return waveforms


def get_radars_waveforms_from_lib():
    """Get listed radar names and listed waveforms from json files.

    Returns
    -------
    radar_lib : dict
        Dictionary of Radars from radarlib.json.
    waveforms : dict
        Dictionary of waveforms from waveformlib.json.
    """
    radar_lib = get_radars()
    waveforms = get_waveforms()
    return radar_lib, waveforms


def get_radars_list(radar_lib=0):
    """Return lists of radar names and radar waveforms from the radar_lib.

    Parameters
    ----------
    radar_lib : dict, optional
        Library of radars. if no input passed, it will read the library from the
        json file.

    Returns
    -------
    radar_names : list of strings.
        List of defined radar names.
    radar_waveforms : list of strings.
        List of waveforms of the defined radars.
    """
    if radar_lib == 0:
        radar_lib = get_radars()
    radar_names = list(radar_lib.values())
    radar_waveforms = list(radar_lib.keys())
    return radar_names, radar_waveforms


def get_waveforms_list(waveforms=0):
    """Return a list of waveforms from the waveforms library.

    Parameters
    ----------
    waveforms : dict, optional
        Library of waveforms. if no input passed, it will read the library from
        the json file.

    Returns
    -------
    waveforms_list : list of strings.
        List of defined waveforms.
    """
    if waveforms == 0:
        waveforms = get_waveforms()
    waveforms_list = []
    for key, waveform in waveforms.items():
        mod_type = waveform['mod_type']
        sweep_dir = waveform['sweep_dir']
        chips = waveform['chips']
        name = generate_waveform_name(mod_type, sweep_dir=sweep_dir, chips=chips)
        waveforms_list.append(name)
    return waveforms_list


def get_radar_name(waveform, radar_lib=0, not_found=RADAR_UNLISTED):
    """Get name of Radar that match the input waveform.

    Parameters
    ----------
    waveform : string
        Name of waveform to look for in the radar library.
    radar_lib : dict, optional
        Library of radars. if no value is passed, it will load from radars json
        file.
    not_found : string, optional
        Name string to return if waveform is not in the library. Default is
        'Unlisted'.

    Returns
    -------
    radar_name : string
        Name of radar that match the waveform. If no match return 'Unlisted'.
    """
    if radar_lib == 0:
        radar_lib = get_radars()
    radar_name = radar_lib.get(waveform, not_found)
    return radar_name


def generate_waveform_name(mod_type, **kwargs):
    """Return waveform name based on the mod_type and parameters.

    Parameters
    ----------
    mod_type : string
        Modulation type of waveform ('Barker', 'LFM').
    Other parameters required according to mod_type:
        'sweep_dir': LFM sweep direction ('up', 'down')
        'chips': Number of chips for Barker code.

    Examples
    --------
    generate_waveform_name('LFM', sweep_dir='up')
    generate_waveform_name('Barker', chips='13')

    Returns
    -------
    name : string
        Waveform name.
    """
    # Define Matched Filter name for plots and classification
    name = ''
    try:
        mod_type = mod_type.upper()
        if mod_type == 'LFM':
            name = 'LFM-' + kwargs['sweep_dir']
        elif mod_type == 'BARKER':
            name = 'Barker' + kwargs['chips']
        else:
            name = 'UNDEFINED'
    except KeyError:
        print('ERROR: Missing parameters for generate_waveform_name')
        name = 'waveform_name_error'
    return name