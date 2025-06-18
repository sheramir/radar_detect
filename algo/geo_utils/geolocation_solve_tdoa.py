"""Geolocation of signal using tdoa method.
Created on Thu Jul 23 09:38:04 2020
@author: v025222357 Amir Sher
"""
import sympy as sy
import numpy as np
from scipy import signal
from scipy.constants import c # speed of light
import matplotlib.pyplot as plt
from utils.string_utils import string_to_num


def get_num_from_solution(sym_solution):
    """Convert sympy solution to a simple number format

    Parameters
    ----------
    sym_solution : sympy solution
        Solution of equation from sympy.

    Returns
    -------
    tuple
        a tuple of numbers from the solution.
    """
    temp_string = str(f'{sym_solution}')
    new_list = string_to_num(temp_string)
    return tuple(new_list)



def tdoa(x, y, ts, showplot=False):
    """Calculate delay (Time Difference of Arrival) between 2 synchronized IQ
    arrays.

    Parameters
    ----------
    x : numpy array
        IQ data from sensor X.
    y : numpy array
        IQ data from sensor Y.
    ts : float
        Sampling time of IQ files (1/fs).
    showplot : boolean, optional
        Show plot of the correlation between IQ arrays. The default is False.

    Returns
    -------
    delay : float
        Time delay between x and y (seconds).
    distance : float
        Calculated distance difference between the sensors.
    """
    if np.iscomplex(x).any():
        x = abs(x)
    if np.iscomplex(y).any():
        y = abs(y)

    x = x - np.mean(x)
    y = y - np.mean(y)

    corr = signal.correlate(x, y, mode='full')
    max_corr = np.argmax(corr)

    delay = (len(x) - max_corr) * ts
    distance = c * delay

    if showplot:
        fig, ax = plt.subplots()
        ax.set_title(f'TDOA Signals Correlation')
        t_axis = np.arange(len(x) - 1, -len(x), -1) * ts
        ax.plot(t_axis, corr / max(corr), 'b')
        ax.grid(True)
        plt.show()

    return delay, distance


def solve_pos_by_tdoa(s_pos, dtoa_list):
    """Use dtoa calculations and sensor positions to calculate emitter position.

    Parameters
    ----------
    s_pos : list of tuples
        List of sensors locations (in UTM-36).
    dtoa_list : list of float
        List of DTOA calculations of the emitter for each sensor pair.

    Returns
    -------
    solution_middle :
        The mean position from the returned calculations (x,y).
    solutions :
        List of all the calculated results.
    """
    num_sensors = len(s_pos)
    num_pairs = int(0.5 * num_sensors * (num_sensors - 1))

    if num_sensors != 3:
        print('ERROR: Should be exactly 3 sensors!')
        return False
    elif num_pairs != len(dtoa_list):
        print('ERROR: Number of sensors does not match DTOA list!')
        return False

    x, y = sy.symbols("x y")
    (x1, y1) = (s_pos[0][0], s_pos[0][1])
    (x2, y2) = (s_pos[1][0], s_pos[1][1])
    (x3, y3) = (s_pos[2][0], s_pos[2][1])

    (dt12, dt13, dt23) = (dtoa_list[0], dtoa_list[1], dtoa_list[2])
    mid_point = tuple(np.average(s_pos, axis=0))

    #r_i = sy.sqrt((x-x_i)**2 + (y-y_i)**2) - dt_ij
    eq12 = sy.sqrt((x - x1)**2 + (y - y1)**2) - sy.sqrt((x - x2)**2 + \
                                                        (y - y2)**2) - dt12
    eq13 = sy.sqrt((x - x1)**2 + (y - y1)**2) - sy.sqrt((x - x3)**2 + \
                                                        (y - y3)**2) - dt13
    eq23 = sy.sqrt((x - x2)**2 + (y - y2)**2) - sy.sqrt((x - x3)**2 + \
                                                        (y - y3)**2) - dt23

    solution12_23 = sy.nsolve((eq12, eq23), (x, y), mid_point)
    solution12_13 = sy.nsolve((eq12, eq13), (x, y), mid_point)
    solution13_23 = sy.nsolve((eq13, eq23), (x, y), mid_point)

    print('location 12_23:', tuple(solution12_23))
    print('location 12_13:', tuple(solution12_13))
    print('location 13_23:', tuple(solution13_23))

    solutions = [get_num_from_solution(solution12_23), get_num_from_solution(
        solution12_13), get_num_from_solution(solution13_23)]
    solution_middle = tuple(np.average(solutions, axis=0))

    return solution_middle, solutions


if __name__ == '__main__':
    s_pos = [(-4, 1), (1, 3), (2, -4)]
    dtoa_list = [0.17, -2.83, -3]
    solve_pos_by_tdoa(s_pos, dtoa_list)