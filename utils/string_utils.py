"""Collection of utilities for strings.
Created on Wed Jul 22 11:37:52 2020
@author: v025222357 Amir Sher
"""
import random
import re


def is_hebrew(s):
    """Check if string contains hebrew letters.

    Parameters
    ----------
    s : string
        Input string to check.

    Returns
    -------
    Boolean
        True if input contains hebrew letters.
    """
    return any("\u0590" < c <= "\u05EA" for c in s)


def reverse_string(s):
    """Return input string in reverse order.

    Parameters
    ----------
    s : string
        Input string.

    Returns
    -------
    string
        Reverse order of the input string.
    """
    return s[::-1]


def fix_hebrew(s):
    """Reverse string if contains hebrew letters, for cases that display hebrew
    in reverse.

    Parameters
    ----------
    s : string
        Input string.

    Returns
    -------
    string
        The input string in reverse order only if contains hebrew (otherwise
        return the original string).
    """
    if is_hebrew(s):
        return reverse_string(s)
    else:
        return s


def str_join(strings, sep='_'):
    """Join multiple strings into one string with a defined seperator.

    Parameters
    ----------
    *strings : strings or a list of strings
        Several strings to join together.
    sep : string, optional
        Seperator that appears between strings. The default is '_'.

    Returns
    -------
    result : string
        A joined string.
    """
    if isinstance(strings, (tuple, list)):
        strings = strings[0] # assuming the first element is the iterable
    result = sep.join(strings)
    return result


def random_string(chars='letters', word_length=4, words=1):
    """Generate random words from a group of chars.

    Chars groups could be:
    - 'lowercase_letters' (or 'lowcase')
    - 'uppercase_letters' (or 'upcase')
    - 'all_letters' (or 'letters')
    - 'numbers' (or 'digits')
    - 'binary'
    - 'signs' (or 'non_letters')
    - 'all_chars' (or 'all')
    - or any characters from a string

    Parameters
    ----------
    chars : string, optional
        Characters group or any user defined string of characters. The default
        is 'letters'.
    word_length : integer, optional
        The length of the generated words. The default is 4.
    words : integer, optional
        Number of words to generate. The default is 1.

    Returns
    -------
    sentence : list of strings
        A list of randomly generated words.
    """
    lowercase_letters = list(range(ord('a'), ord('z') + 1))
    uppercase_letters = list(range(ord('A'), ord('Z') + 1))
    all_letters = lowercase_letters + uppercase_letters
    numbers = list(range(ord('0'), ord('9') + 1))
    binary = list(range(ord('0'), ord('1') + 1))
    signs = list(range(91, 97)) + list(range(33, 48))
    all_chars = all_letters + numbers + signs

    low_letters_lst = ['lowercase_letters', 'lowercase', 'lowcase',
                       'lowcase letters', 'low case']
    up_letters_lst = ['uppercase_letters', 'uppercase', 'upcase',
                     'upcase_letters', 'up_case']
    all_letters_lst = ['all_letters', 'letters']
    numbers_lst = ['numbers', 'all_numbers', 'digits']
    binary_lst = ['binary']
    signs_lst = ['signs', 'non_letters']
    all_chars_lst = ['all_chars', 'all']

    if chars in low_letters_lst:
        char_group = lowercase_letters
    elif chars in up_letters_lst:
        char_group = uppercase_letters
    elif chars in all_letters_lst:
        char_group = all_letters
    elif chars in numbers_lst:
        char_group = numbers
    elif chars in binary_lst:
        char_group = binary
    elif chars in signs_lst:
        char_group = signs
    elif chars in all_chars_lst:
        char_group = all_chars
    else:
        char_group = [ord(i) for i in chars]

    sentence = []
    for word in range(words):
        word = ''.join(chr(random.choice(char_group)) for i in range(word_length))
        sentence.append(word)
    return sentence


def string_to_num(s):
    """Extract numbers from a string.

    Example: 'My age is 40 and I am 1.87m tall.' -> [40.0, 1.87]

    Parameters
    ----------
    s : string
        Input string.

    Returns
    -------
    num_list : list of float
        List of float numbers extracted from the string.
    """
    str_list = re.findall(r'\d+\.?\d*', s)
    num_list = [float(i) for i in str_list]
    return num_list