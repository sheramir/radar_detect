"""Utility functions for numbers presentation."""
import re
import numpy as np
import numbers


def format_num(num, unit='', precision=3):
    """Convert number to a short format string or convert string to number.

    For example: 3400 => '3.4k', 4e-05=>'40u'
    '1.2M' => 1200000.0
    '3k' => 3000.0
    '50%' => 0.5
    format_num(0.5, unit='') => '50%'
    format_num(2, unit='db') => '3.01dB'
    format_num('3.01db') => 2.0

    Parameters
    ----------
    num : int/ float / string / list / ndarray
        The number (or list/array of numbers) to convert.
    unit : string, optional
        Unit string (for example 'Hz') The default is none.
        If unit is '' the num will be converted to percentage (100)
    precision : integer, optional
        number of digits after the period. The default is 3.

    Returns
    -------
    num_str : string / float (or list of string/float)
        Formatted number (or list of formatted number).
    """
    prefix = 'pnum kMGTP' # e-12..e+15
    if isinstance(num, numbers.Number):
        n_str = ''
        sign = 1
        if unit == '':
            num = round(num * 100, precision)
            n_str = str(int(num)) if int(num) == num else str(num)
            unit = '%'
        elif unit.upper() == 'DB':
            num = round(10 * np.log10(num), precision)
            n_str = str(num)
            unit = 'dB'
        elif not num == 0:
            n = float(num)
            if n < 0:
                sign = -1
                n = abs(n)
            for i in range(len(prefix)):
                dim = (i - 4) * 3
                lim = 10**dim
                nextlim = 10**(dim + 3)
                if i == (len(prefix) - 1):
                    nextlim = float('inf')
                if (n >= lim) and (n < nextlim):
                    n_new = n / lim
                    n_new = round(n_new * 10**precision) / 10**precision
                    if n_new.is_integer():
                        n_new = int(n_new) # remove dot
                    n_str = str(n_new)
                    unit = prefix[i] + unit
                    n_str = n_str.replace(' ', '') # remove spaces
                    num_str = (f'{sign}' if sign == -1 else '') + n_str + unit
                    return num_str
        num_str = (f'{sign}' if sign == -1 else '') + n_str + unit
        return num_str
    elif isinstance(num, str):
        num = num.replace(' ', '') # remove spaces
        num = num.replace('K', 'k') # in case K is used instead k
        #get pure number string
        pure_num = (re.split('[A-Z]|[a-z]', num)[0])
        num_prefix = num[len(pure_num):] if len(num) > len(pure_num) else ''
        if num_prefix in prefix and num_prefix != '':
            dim = 3 * (prefix.find(num_prefix) - 4)
        elif num_prefix == '':
            dim = -2 # this assumes percentage input
        elif num_prefix.upper() == 'DB':
            return round(10**(float(pure_num) / 10), precision)
        else:
            dim = 0
        try:
            num_float = float(pure_num)
            return num_float * 10**dim
        except ValueError:
            return None
    elif isinstance(num, list) or isinstance(num, np.ndarray):
        return [format_num(num_element, unit=unit, precision=precision) for num_element in num]
    return num


def num_round(num, precision):
    """Round a number by precision.

    For example: num_round(234,100) = 200
    num_round(34234, '1k') = 34000.0
    num_round('39234','10k') = 40000

    Parameters
    ----------
    num : integer or float or string
        The number to perform the rounding function.
    precision : integer or float or string
        The precision of the output number. for example: 1000 or '1k'.
        precision should be lower than num.

    Returns
    -------
    float or int
        The rounded number.
    """
    if isinstance(precision, str):
        precision = format_num(precision)
    if isinstance(num, str):
        num = format_num(num)
    return precision * round(num / precision)