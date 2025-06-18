"""Find overlapping XDAT files triplets, according to XHDR gps capture time.
Created on Tue Jun  9 13:17:05 2020
@author: v025222357 Amir Sher
"""
import numpy as np
import glob
from datetime import datetime
from datetime import timedelta
import math
import os
from algo.RF_Classes.IQData import IQData as IQData
from utils import xhdr_utils as xhdr_utils


def get_xhdr(file):
    sample_rate, result = xhdr_utils.get_sample_rate(file)
    samples, result = xhdr_utils.get_samples(file)
    samples = int(float(samples))
    sample_rate = int(float(sample_rate))
    length = samples * 1 / sample_rate
    start_capture_str, result = xhdr_utils.get_start_capture(file)
    (y, j, h, m, s) = start_capture_str.split(':')
    (s1, s2) = s.split('.')
    s = s1 + '.' + s2[0:6]
    dt_string = y + ':' + (str(int(j))) + ':' + h + ':' + m + ':' + s
    start_time = datetime.strptime(dt_string, "%Y:%j:%H:%M:%S.%f")
    end_time = start_time + timedelta(seconds=length)
    return start_time, end_time, length


def overlap_time(start1, start2, end1, end2):
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = timedelta_to_seconds(overlap_end - overlap_start)
    overlap = True if overlap_length > 0 else False
    return overlap, overlap_start, overlap_end, overlap_length


def timedelta_to_seconds(timedeltaobj):
    time_in_seconds = timedeltaobj.seconds + \
                      1e-6 * timedeltaobj.microseconds + \
                      timedeltaobj.days * 60 * 60 * 24
    return time_in_seconds


def break_full_length(length, max_length, min_length, max_sections, start):
    sections = math.floor(length / max_length)
    if sections > max_sections:
        sections = max_sections
    if sections > 0:
        section_length = max_length
        step = length / sections
        starts = [0.0] * sections
        for i in range(sections):
            starts[i] = start + timedelta(seconds=i * step)
    elif sections == 0:
        if length > min_length:
            section_length = length
            starts = [start]
        else:
            section_length = 0
            starts = []
    return starts, section_length


def convert_xhdr_to_xdat(file):
    file = file[:-5] + '.xdat'
    return file


def str_concat(strings, sep='_'):
    result = ""
    for in_str in strings:
        result += str(in_str) + sep
    result = result[:-len(sep)]
    return result


source_base = r'Z:\REPOSITORY_ORIGINAL\s-band'
dest_base = r'C:\IQ_Data\Radar\sync_sensors\triplets2'

kisufim = ('Kisufim', r'כיסופים')
nahaloz = ('Nahaloz', r'נחל עוז')
yiftah = ('Yiftah', r'יפתח')

sensors = (kisufim, nahaloz, yiftah)

source_kisufim = os.path.join(source_base, kisufim[1])
source_nahaloz = os.path.join(source_base, nahaloz[1])
source_yiftah = os.path.join(source_base, yiftah[1])

radar_kipa = ('Kipa', 'כיפת ברזל')
radar_rada = ('Rada', 'ראדה')
radar_kipa_rada = ('Kipa+Rada', 'כיפת ברזל ראד')
radars = (radar_kipa, radar_rada, radar_kipa_rada)

MAX_LENGTH = 100e-3 # maximum file length in seconds
MIN_LENGTH = 20e-3 # minimum file length
MAX_FILES_FROM_ORIG = 5 # maximum generated files from one pair


xhdr_dict = {radar_kipa[0]: {
    kisufim[0]: {
        'files': [],
        'start': [],
        'end': [],
        'length': []},
    nahaloz[0]: {
        'files': [],
        'start': [],
        'end': [],
        'length': []},
    yiftah[0]: {
        'files': [],
        'start': [],
        'end': [],
        'length': []}
},
    radar_rada[0]: {
        kisufim[0]: {
            'files': [],
            'start': [],
            'end': [],
            'length': []},
        nahaloz[0]: {
            'files': [],
            'start': [],
            'end': [],
            'length': []},
        yiftah[0]: {
            'files': [],
            'start': [],
            'end': [],
            'length': []}
},
    radar_kipa_rada[0]: {
        kisufim[0]: {
            'files': [],
            'start': [],
            'end': [],
            'length': []},
        nahaloz[0]: {
            'files': [],
            'start': [],
            'end': [],
            'length': []},
        yiftah[0]: {
            'files': [],
            'start': [],
            'end': [],
            'length': []}
}}

#measure length and capturing time for each file in repository
print(f'Read xhdr files from {source_base}.')
ri = -1
for radar in radars:
    ri += 1
    radar_kisufim_path = os.path.join(source_base, kisufim[1], radar[1])
    radar_nahaloz_path = os.path.join(source_base, nahaloz[1], radar[1])
    radar_yiftah_path = os.path.join(source_base, yiftah[1], radar[1])

    kisufim_files = glob.glob(os.path.join(radar_kisufim_path, '*.xhdr'))
    nahaloz_files = glob.glob(os.path.join(radar_nahaloz_path, '*.xhdr'))
    yiftah_files = glob.glob(os.path.join(radar_yiftah_path, '*.xhdr'))

    xhdr_dict[radar[0]][kisufim[0]]['files'] = kisufim_files
    xhdr_dict[radar[0]][nahaloz[0]]['files'] = nahaloz_files
    xhdr_dict[radar[0]][yiftah[0]]['files'] = yiftah_files

    for kis_file in kisufim_files:
        start_time, end_time, length = get_xhdr(kis_file)
        xhdr_dict[radar[0]][kisufim[0]]['start'].append(start_time)
        xhdr_dict[radar[0]][kisufim[0]]['end'].append(end_time)
        xhdr_dict[radar[0]][kisufim[0]]['length'].append(length)

    for nahal_file in nahaloz_files:
        start_time, end_time, length = get_xhdr(nahal_file)
        xhdr_dict[radar[0]][nahaloz[0]]['start'].append(start_time)
        xhdr_dict[radar[0]][nahaloz[0]]['end'].append(end_time)
        xhdr_dict[radar[0]][nahaloz[0]]['length'].append(length)

    for yiftah_file in yiftah_files:
        start_time, end_time, length = get_xhdr(yiftah_file)
        xhdr_dict[radar[0]][yiftah[0]]['start'].append(start_time)
        xhdr_dict[radar[0]][yiftah[0]]['end'].append(end_time)
        xhdr_dict[radar[0]][yiftah[0]]['length'].append(length)


#Find overlap files in data
print(f'Write xhdr/xdat files to {dest_base}:')
num = -1
for radar, data in xhdr_dict.items():
    ki = -1
    for kis_file in data[kisufim[0]]['files']:
        ki += 1
        start1 = data[kisufim[0]]['start'][ki]
        end1 = data[kisufim[0]]['end'][ki]
        length1 = data[kisufim[0]]['length'][ki]

        ni = -1
        for nahal_file in data[nahaloz[0]]['files']:
            ni += 1
            start2 = data[nahaloz[0]]['start'][ni]
            end2 = data[nahaloz[0]]['end'][ni]
            length2 = data[nahaloz[0]]['length'][ni]

            is_overlap, ov_start, ov_end, ov_length = \
                overlap_time(start1, start2, end1, end2)
            ov_length = round(ov_length * 1000 * 10) / 1000

            if is_overlap and ov_length > MIN_LENGTH:
                yi = -1
                for yiftah_file in data[yiftah[0]]['files']:
                    yi += 1
                    start3 = data[yiftah[0]]['start'][yi]
                    end3 = data[yiftah[0]]['end'][yi]
                    length3 = data[yiftah[0]]['length'][yi]

                    is_overlap3, ov_start3, ov_end3, ov_length3 = \
                        overlap_time(ov_start, start3, ov_end, end3)
                    ov_length3 = round(ov_length3 * 1000 * 10) / 1000

                    if is_overlap3 and ov_length3 > MIN_LENGTH:
                        starts, ov_length = break_full_length(ov_length3,
                                                              MAX_LENGTH, MIN_LENGTH,
                                                              MAX_FILES_FROM_ORIG,
                                                              ov_start3)
                        print(kis_file)
                        print(nahal_file)
                        print(yiftah_file)
                        print(f'kisufim length = {length1}')
                        print(f'nahal length = {length2}')
                        print(f'yiftah length = {length3}')
                        print(f'overlap length = {ov_length3}')

                        for start in starts:
                            num += 1
                            num_str = ('000' + str(num))[-3:]
                            file_date_str = start.strftime("%d%m%Y_%H%M%S")

                            dst_name_kisufim = str_concat([num_str, kisufim[0],
                                                           radar, file_date_str])
                            dst_name_nahaloz = str_concat([num_str, nahaloz[0],
                                                           radar, file_date_str])
                            dst_name_yiftah = str_concat([num_str, yiftah[0],
                                                          radar, file_date_str])

                            dst_name_kisufim = os.path.join(dest_base,
                                                            dst_name_kisufim)
                            dst_name_nahaloz = os.path.join(dest_base,
                                                            dst_name_nahaloz)
                            dst_name_yiftah = os.path.join(dest_base,
                                                           dst_name_yiftah)

                            print(dst_name_kisufim)
                            print(dst_name_nahaloz)
                            print(dst_name_yiftah)

                            kisufim_start = timedelta_to_seconds(start - start1)
                            nahal_start = timedelta_to_seconds(start - start2)
                            yiftah_start = timedelta_to_seconds(start - start3)

                            print(f'kisufim start {kisufim_start}')
                            print(f'nahal start {nahal_start}')
                            print(f'yiftah start {yiftah_start}')

                            iqdata1 = IQData(kis_file, kisufim_start, ov_length)
                            iqdata1.save_xdat(dst_name_kisufim)
                            name = os.path.basename(dst_name_kisufim)[:-5]
                            iqdata1.plot_iq(save=True, file=name, path=dest_base,
                                            plot1='time.abs', plot2='psd')

                            iqdata2 = IQData(nahal_file, nahal_start, ov_length)
                            iqdata2.save_xdat(dst_name_nahaloz)
                            name = os.path.basename(dst_name_nahaloz)[:-5]
                            iqdata2.plot_iq(save=True, file=name, path=dest_base,
                                            plot1='time.abs', plot2='psd')

                            iqdata3 = IQData(yiftah_file, yiftah_start, ov_length)
                            iqdata3.save_xdat(dst_name_yiftah)
                            name = os.path.basename(dst_name_yiftah)[:-5]
                            iqdata3.plot_iq(save=True, file=name, path=dest_base,
                                            plot1='time.abs', plot2='psd')