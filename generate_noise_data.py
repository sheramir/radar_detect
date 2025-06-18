"""Create collection of XDAT files with augmented synthetic noise.

Created on Sun Jun  7 12:52:16 2020

@author: v025222357 Amir Sher
"""
from algo.RF_Classes.IQData import IQData as IQData
from utils import num_utils as nu
import os
import glob
import gc


def str_concat(strings, sep='_'):
    result = ""
    for in_str in strings:
        result += str(in_str) + sep
    result = result[:-len(sep)]
    return result


source_path = r'Z:\Testset\Noise\source'
dest_path = r'Z:\Testset\Noise\files'

snr_list = ['20db', '15db', '10db', '7db', '5db', '3db']
relative_noise_list = [10, 5, 3, 2]

orig_index = 0
snr_index = 1
relative_index = 2

num = -1
xdat_files = glob.glob(os.path.join(source_path, '*.xdat'))
num_files = len(xdat_files)

for xdat_file in xdat_files:
    num += 1
    str_num = ('000' + str(num))[-3:]
    base_name = os.path.split(xdat_file)[1]
    index_num, sensor, radar = base_name.split('_')[0:3]

    dest_base_name = str_concat([sensor, radar, index_num])

    print(f'Parsing file {num} of {num_files}: {xdat_file}.')
    #Execute the main function
    my_IQ = IQData(xdat_file)

    new_name = str_concat([str_num, dest_base_name, orig_index, 'orig'])
    dest_name = os.path.join(dest_path, new_name)
    my_IQ.save_xdat(dest_name)
    print(dest_name)

    for snr in snr_list:
        num += 1
        str_num = ('000' + str(num))[-3:]
        new_name = str_concat([str_num, dest_base_name, snr_index, snr])
        dest_name = os.path.join(dest_path, new_name)
        print(dest_name)
        new_IQ = my_IQ.copy()
        new_IQ.generate_awgn(snr, method='snr')
        new_IQ.save_xdat(dest_name)
        del (new_IQ)

    for rel_noise in relative_noise_list:
        num += 1
        str_num = ('000' + str(num))[-3:]
        new_name = str_concat([str_num, dest_base_name, relative_index, rel_noise])
        dest_name = os.path.join(dest_path, new_name)
        print(dest_name)
        new_IQ = my_IQ.copy()
        new_IQ.generate_awgn(rel_noise, method='relative')
        new_IQ.save_xdat(dest_name)
        del (new_IQ)

    gc.collect()

    #power, rms = my_IQ.get_signal_power(showplot=True)
    #print(f'signal power {nu.format_num(power)}. RMS={nu.format_num(rms)}')

    #calc_noise1, _ = my_IQ.get_noise_level(showplot=False)

    #new_noise, current_noise = my_IQ.generate_awgn('3db', method='snr')
    #print(f'new noise is {nu.format_num(new_noise)} and current noise is {nu.format_num(current_noise)}')

    #calc_noise2, _ = my_IQ.get_noise_level(showplot=False)
    #print(f'new calc noise is {nu.format_num(calc_noise2)}')