"""Utilities for creating, using and saving pandas dataframes."""
import numpy as np
import pandas as pd
from os import path


def get_unique_column_values(my_dataframe, column):
    """Get sorted unique values from a dataframe column.

    Parameters
    ----------
    my_dataframe : dataframe
        pandas dataframe.
    column : string
        The name of the column.

    Returns
    -------
    sorted_values : numpy array
        Sorted array of unique values from the column.
    """
    unique_values = my_dataframe[column].unique()
    sorted_values = np.sort(list(unique_values))
    return sorted_values


def get_sub_headers(main_header, sub_header):
    """Generate headers and sub-headers for creation of dataframe with
    sub-headers.

    Parameters
    ----------
    main_header : list or array
        Array of main header labels.
    sub_header : list or array
        Array of sub-header labels.

    Returns
    -------
    main_header_list : list
        List of main header labels in the size of main-header X sub-header.
    sub_header_list : list
        List of sub-header labels in the size of main-header X sub-header.
    """
    main_header_len = len(main_header)
    sub_header_len = len(sub_header)
    if isinstance(main_header, np.ndarray):
        main_header = main_header.tolist()
    if isinstance(sub_header, np.ndarray):
        sub_header = sub_header.tolist()
    main_header_list = main_header * sub_header_len
    main_header_list.sort()
    sub_header_list = sub_header * main_header_len
    return main_header_list, sub_header_list


def create_dataframe(index, columns):
    """Create new dataframe from index and columns lists.

    Parameters
    ----------
    index : list
        List of index labels.
    columns : list or tuple of (header, sub-header)
        Column lables for main header and sub-header.

    Returns
    -------
    dataframe
        Dataframe with zeros values in the size of index X columns.
    """
    if isinstance(columns, tuple): #dataframe with sub-headers
        columns = pd.MultiIndex.from_tuples(zip(columns[0], columns[1]))
    indx_len = len(index)
    col_len = len(columns)
    return pd.DataFrame(np.zeros((indx_len, col_len)), index=index, columns=
                        columns)


def remove_zero_columns(my_df):
    """Remove zero values columns from dataframe.

    Parameters
    ----------
    my_df : dataframe
        Dataframe from which the empty columns will be removed.

    Returns
    -------
    df : dataframe
        Dataframe without empty (zeros) columns.
    """
    for cols in my_df.columns:
        if my_df[cols].sum().sum() == 0:
            my_df = my_df.drop([cols], axis=1)
    return my_df


def df_to_excel(file_name, dataframes, file_suffix='', sheet_names=[]):
    """Write dataframes to excel file in different sheets.
    If the file name already exists, a number will be added automatically to
    the new file name.

    Parameters
    ----------
    file_name : string
        Path file name destination of output file.
    file_suffix : string, Optional
        Suffix to be added to the output file name. Default is ''.
    dataframes : list of dataframes
        List of dataframes to write into excel file (in each sheet).
    sheet_names : list of strings, Optional.
        Name of sheets for dataframes. If list is partial or not included, name
        will be generated.

    Returns
    -------
    None.
    """
    excel_file = file_name + file_suffix
    if excel_file[-5:] == '.xlsx':
        excel_file = excel_file[0:-5] # remove extension
    excel_file_no_num = excel_file
    file_count = 0
    while path.exists(excel_file + '.xlsx'): # if exists create a new name
        file_count += 1
        excel_file = f'{excel_file_no_num}({file_count})'
    excel_file = excel_file + '.xlsx'

    sheet = 0
    with pd.ExcelWriter(excel_file) as writer:
        for df in dataframes:
            sheet += 1
            if len(sheet_names) < sheet:
                sheetname = 'sheet' + str(sheet)
            else:
                sheetname = sheet_names[sheet - 1]
            df.to_excel(writer, sheet_name=sheetname)


def df_to_csv(path_name, dataframes, csv_names=[]):
    """Write dataframes to CSV files.
    If the file names already exists, a number will be added automatically to
    the new files names.

    Parameters
    ----------
    path_name : string
        Path file name destination of output file.
    dataframes : list of dataframes
        List of dataframes to write into CSV files.
    csv_names : list of strings, Optional.
        Name of CSV files. If list is partial or not included, name will be
        generated.

    Returns
    -------
    None.
    """
    sheet = 0
    file_count = 0
    count_str = ''
    csv_files = []
    for df in dataframes:
        sheet += 1
        if len(csv_names) < sheet:
            sheetname = 'csv' + str(sheet)
        else:
            sheetname = csv_names[sheet - 1]
        if sheetname[-4:] == '.csv':
            sheetname = sheetname[0:-4]
        csv_files.append(path.join(path_name, sheetname))
        while path.exists(csv_files[sheet - 1] + count_str + '.csv'):
            file_count += 1
            count_str = f'({str(file_count)})'

    sheet = 0
    for df in dataframes:
        csv_name = csv_files[sheet] + count_str + '.csv'
        df.to_csv(csv_name)
        sheet += 1