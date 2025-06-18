"""Utility functions for reading and writing xhdr and xdat files."""
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np
from datetime import datetime
from consts import *


def get_xhdr_root(xhdr_link):
    """Read xhdr file and return the xml root.

    Parameters
    ----------
    xhdr_link : string
        xhdr file path and name.

    Returns
    -------
    root : xml root
        The root of the xhdr file or None if unable to reach file
    """
    if "http" in xhdr_link:
        try:
            data = urlopen(xhdr_link)
        except HTTPError as err:
            print(f'xhdr validator: kube error when reach {xhdr_link}: {err}')
            return None
        root = ET.fromstring(data.read())
    else: # local file
        root = ET.parse(xhdr_link).getroot()
    return root


def get_and_validate_capture_xml(root):
    """Get and vaildate the 'Capture' xhdr branch.

    Parameters
    ----------
    root : xml root
        The root of the xhdr file.

    Returns
    -------
    capture : TYPE
        The 'capture' branch of the xhdr file or None if fail.
    result : boolean
        return success or fail.
    """
    result = None
    captures = root.find(CAPTURES)
    if captures is None:
        result = 'element not found (CAPTURES)'
        print('xhdr validator: kube error: ', result)
        return None, result
    capture = captures.find(CAPTURE)
    if capture is None:
        result = 'element not found (CAPTURE)'
        print('xhdr validator: kube error: ', result)
        return capture, result


def get_start_capture(xhdr_link):
    """Get 'start_capture' property from the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.

    Returns
    -------
    string
        Value of 'start_capture' property.
    """
    return get_capture_property(xhdr_link, START_CAPTURE)


def get_acq_scale_factor(xhdr_link):
    """Get 'acq_scale_factor' property from the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.

    Returns
    -------
    string
        Value of 'acq_scale_factor' property.
    """
    return get_capture_property(xhdr_link, 'acq_scale_factor')


def get_sample_rate(xhdr_link):
    """Get 'sample_rate' property from the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.

    Returns
    -------
    string
        Value of 'sample_rate' property.
    """
    return get_capture_property(xhdr_link, 'sample_rate')


def get_center_frequency(xhdr_link):
    """Get 'center_frequency' property from the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.

    Returns
    -------
    integer
        Value of 'center_frequency' property.
    result : boolean
        A success flag.
    """
    freq, result = get_capture_property(xhdr_link, 'center_frequency')
    return int(float(freq)), result


def get_samples(xhdr_link):
    """Get 'samples' property from the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.

    Returns
    -------
    string
        Value of 'samples' property.
    """
    return get_capture_property(xhdr_link, 'samples')


def get_acquisition_bandwidth(xhdr_link):
    """Get the 'acquisition_bandwidth' or 'span' property from the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.

    Returns
    -------
    integer
        Value of acquisition_bandwidth (or span).
    result : boolean
        A success flag.
    """
    val, result = get_capture_property(xhdr_link, 'acquisition_bandwidth')
    # Different sensors use 'span' or 'acquisition_bandwidth'
    if val is None:
        val, result = get_capture_property(xhdr_link, 'span')

    if val is None:
        print('xhdr validator: kube error:', result)
        val = 0
    return int(float(val)), result


def get_capture_property(xhdr_link, property_name):
    """Get a property from the 'capture' branch of the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.
    property_name : string
        The name of the property.

    Returns
    -------
    string
        The propery value.
    boolean
        A success flag.
    """
    root = get_xhdr_root(xhdr_link)
    capture, result = get_and_validate_capture_xml(root)
    if result is not None:
        return None, result

    capture_property = capture.get(property_name)
    if capture_property is None:
        result = element_not_found(property_name)
        return None, result
    return capture_property, None


def get_data_property(xhdr_link, property_name):
    """Get a property from the 'data' branch of the xhdr file.

    Parameters
    ----------
    xhdr_link : string
        Path and name of the xhdr file.
    property_name : string
        The name of the property.

    Returns
    -------
    string
        The propery value.
    boolean
        A success flag.
    """
    root = get_xhdr_root(xhdr_link)
    data, result = get_and_validate_data_xml(root)
    if result is not None:
        return None, result
    data_property = data.get(property_name)
    if data_property is None:
        result = element_not_found(property_name)
        print('xhdr validator: kube error:', result)
        return None, result
    return data_property, None


def get_and_validate_data_xml(root):
    """Get and vaildate the 'Data' xhdr branch.

    Parameters
    ----------
    root : xml root
        The root of the xhdr file.

    Returns
    -------
    data
        The 'data' branch of the xhdr file.
    result : boolean
        return success or fail.
    """
    result = None
    data_files = root.find(DATA_FILES)
    if data_files is None:
        result = element_not_found(DATA_FILES)
        print('xhdr validator: kube error: ', result)
        return None, result
    data = data_files.find(DATA)
    if data is None:
        result = element_not_found(DATA)
        print('xhdr validator: kube error: ', result)
        return data, result


def element_not_found(property_name):
    return f'{property_name} element not found in xml'


def get_complex_array(acq_scale_factor, dt, input_file, length,
                      starting_point):
    """Get IQ complex array from xdat file.

    Parameters
    ----------
    acq_scale_factor : float
        Scaling factor.
    dt : float
        Time steps of the sampled data.
    input_file : string
        Path of xdat input file.
    length : float
        Length of time of required samples from the data (in seconds).
    starting_point : float
        Time of first sample (in seconds).

    Returns
    -------
    Complex array
        Complex array of IQ data.
    """
    start = 2 * int(starting_point / dt)
    end = 2 * int(starting_point / dt) + 2 * int(length / dt)
    num_samples = 2 * int(length / dt)
    offset_bytes = start * 2
    #IQ = np.fromfile(input_file, dtype='int16')[start:end]
    IQ = np.fromfile(input_file, dtype='int16', count=num_samples, offset=
                     offset_bytes) # faster way
    scaled_array = IQ * acq_scale_factor / 65536
    return scaled_array[0::2] + 1j * scaled_array[1::2]


def save_xdat(output_file, iq_data, acq_scale_factor):
    """Save complex IQ data to xdat file.

    Parameters
    ----------
    output_file : string
        Name and path of the output xdat file (could be with or without 'xdat'
        extension).
    iq_data : complex numpy array
        The complex array of IQ data to be written to file.
    acq_scale_factor : float
        A scaling factor (as found in the xhdr file).

    Returns
    -------
    None.

    """
    if (output_file[-5:] == '.xdat') or (output_file[-5:] == '.xhdr'):
        name = output_file[:-5]
    else:
        name = output_file
    output_file = name + '.xdat'
    scale = acq_scale_factor / 65536
    re = iq_data.real / scale
    im = iq_data.imag / scale
    arr = np.array([val for pair in zip(re, im) for val in pair], dtype='int16')
    arr.tofile(output_file)


def save_xhdr(output_file, acq_scale_factor, center_frequency,
              sample_rate, span, start_capture, samples):
    """Save xhdr data to new file.

    Parameters
    ----------
    output_file : string
        Name and path of the output xhdr file (could be with or without 'xhdr'
        extension).
    acq_scale_factor : float
        Scaling factor.
    center_frequency : integer/ float
        Center frequency property (in Hz).
    sample_rate : integer
        Sample rate property.
    span : integer
        File span property (in Hz).
    start_capture : string or datetime object
        Time of start of capture (date and exact time).
    samples : integer
        Samples property.

    Returns
    -------
    None.

    """
    if (output_file[-5:] == '.xdat') or (output_file[-5:] == '.xhdr'):
        name = output_file[:-5]
    else:
        name = output_file
    output_file = name + '.xhdr'

    if isinstance(start_capture, datetime):
        day = str(int(start_capture.strftime("%j")) - 1)
        day = ('00' + day)[-3:]
        str_start_capture = start_capture.strftime(f"%Y:{day}:%H:%M:%S.%f")
    else:
        str_start_capture = start_capture

    xhdr_header = f'<xcom_header header_version="1.0" name="{name}.xhdr" ' \
                  f'sw_version="1.1.0.0">\n'
    xhdr_capture = f'<capture acq_scale_factor="{acq_scale_factor}" ' \
                   f'center_frequency="{int(center_frequency)}" name="{name}" ' \
                   f'sample_rate="{int(sample_rate)}" span="{int(span)}" ' \
                   f'start_capture="{str_start_capture}" />\n'
    xhdr_data = f'<data channel_count="1" data_encoding="int16" ' \
                f'iq_interleave="true" little_endian="true" ' \
                f'name="{name}.xdat" protected="false" sample_resolution="16" ' \
                f'samples="{int(samples)}" signed_type="true" />\n'
    xhdr_marker = f'<marker count="1" format="XML" name="{name}.xmrk" />\n'

    with open(output_file, 'w') as xhdr_file:
        xhdr_file.write(xhdr_header)
        xhdr_file.write('<captures>\n')
        xhdr_file.write(xhdr_capture)
        xhdr_file.write('</captures>\n')
        xhdr_file.write('<data_files>\n')
        xhdr_file.write(xhdr_data)
        xhdr_file.write('</data_files>\n')
        xhdr_file.write('<marker_files>\n')
        xhdr_file.write(xhdr_marker)
        xhdr_file.write('</marker_files>\n')
        xhdr_file.write('</xcom_header>\n')