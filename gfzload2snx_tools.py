import numpy as np
import scipy.signal as signal
import os
from geodezyx import utils
from geodezyx import conv

def get_model_files(model_descriptor):
    """
    Maps a model identifier to the corresponding model type.

    Parameters:
    model_descriptor (str): Model descriptor containing identifiers.

    Returns:
    list: List of model types.
    """
    mapper = {'A': 'ntal', 'O': 'ntol', 'S': 'slel', 'H': 'cwsl'}
    model = [mapper[x] for x in model_descriptor]
    return model

def get_model_type(filename):
    """
    Determines the model type based on the filename.

    Parameters:
    filename (str): Name of the file.

    Returns:
    str: Model type identifier.
    """
    if '.ntal.' in filename:
        return 'A'
    elif '.ntol.' in filename:
        return 'O'
    elif '.slel.' in filename:
        return 'S'
    elif '.cwsl.' in filename:
        return 'H'
    else:
        return 'Unknown'

def get_unique_prefixes(directory):
    """
    Retrieves unique file prefixes from a directory.

    Parameters:
    directory (str): Path to the directory.

    Returns:
    set: Unique prefixes from the files in the directory.
    """
    files = os.listdir(directory)
    prefixes = [file[:4] for file in files if os.path.isfile(os.path.join(directory, file))]
    unique_prefixes = set(prefixes)
    return unique_prefixes

def load_files(file_paths):
    """
    Loads data files and returns a dictionary of DataFrames.

    Parameters:
    file_paths (list): List of file paths to load.

    Returns:
    dict: Dictionary with keys as station and model type, and values as DataFrames.
    """
    dataframes = {}
    for file_path in file_paths:
        df = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None)
        df.columns = ['yyyy', 'mm', 'dd', 'hh', 'R', 'EW', 'NS']
        df['datetime'] = pd.to_datetime(df[['yyyy', 'mm', 'dd', 'hh']].astype(str).agg('-'.join, axis=1),
                                        format='%Y-%m-%d-%H')
        df.drop(columns=['yyyy', 'mm', 'dd', 'hh'], inplace=True)
        df.set_index('datetime', inplace=True)
        daily_mean_df = df.resample('D').mean()
        station = os.path.basename(file_path)[:4]
        model_type = get_model_type(file_path)
        key = f"{station}_{model_type}"
        dataframes[key] = daily_mean_df
    return dataframes

def high_pass_filter(data, column, cutoff_days):
    """
    Applies a high-pass filter to a specific column in a DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data.
    column (str): Name of the column to apply the filter to.
    cutoff_days (int): Cutoff frequency in days.

    Returns:
    np.ndarray: Filtered data.
    """
    cutoff_freq = 1 / cutoff_days
    (b, a) = signal.butter(1, cutoff_freq, btype='high', fs=1)
    filtered_data = signal.filtfilt(b, a, data[column])
    return filtered_data

def filter_frequencies(dataframes, cutoff_days=500):
    """
    Filters frequencies from the dataframes using a high-pass filter.

    Parameters:
    dataframes (dict): Dictionary of DataFrames to filter.
    cutoff_days (int): Cutoff frequency in days.

    Returns:
    dict: Dictionary of filtered DataFrames.
    """
    filtered_dataframes = {}
    for key, df in dataframes.items():
        filtered_df = df.copy()
        for column in df.columns:
            if column in ['R', 'EW', 'NS']:
                filtered_df[column] = high_pass_filter(df, column, cutoff_days)
        filtered_dataframes[key] = filtered_df
    return filtered_dataframes

def combine_selected_files(filtered_dataframes):
    """
    Combines selected filtered DataFrames into a single DataFrame.

    Parameters:
    filtered_dataframes (dict): Dictionary of filtered DataFrames.

    Returns:
    tuple: Combined DataFrame and solution name.
    """
    combined_df = sum(filtered_dataframes.values())
    solution_name = ''.join(sorted([x.split('_')[-1] for x in list(filtered_dataframes.keys())]))
    return combined_df, solution_name

def save_combined_data(combined_df, output_file):
    """
    Saves the combined DataFrame to a file.

    Parameters:
    combined_df (pd.DataFrame): Combined DataFrame.
    output_file (str): Path to the output file.
    """
    combined_df.to_pickle(output_file)

def ecef_to_geodetic(x, y, z):
    """
    Converts ECEF coordinates to geodetic coordinates (latitude, longitude, height).

    Parameters:
    x, y, z (float): ECEF coordinates.

    Returns:
    tuple: Latitude and longitude in radians.
    """
    a = 6378137.0  # Semi-major axis
    e2 = 6.69437999014e-3  # Square of eccentricity

    b = np.sqrt(a ** 2 * (1 - e2))  # Semi-minor axis
    ep = np.sqrt((a ** 2 - b ** 2) / b ** 2)
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * a, p * b)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep ** 2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)

    return lat, lon

def topo_to_ecef(lat, lon, NS, EW, R):
    """
    Converts topocentric displacements (NS, EW, R) to ECEF frame.

    Parameters:
    lat (float): Latitude of the station in radians.
    lon (float): Longitude of the station in radians.
    NS (float): Displacement in the North-South direction.
    EW (float): Displacement in the East-West direction.
    R (float): Displacement in the Up (Radial) direction.

    Returns:
    tuple: Displacements in the ECEF frame (dX, dY, dZ).
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    T = np.array([
        [-sin_lat * cos_lon, -sin_lon, cos_lat * cos_lon],
        [-sin_lat * sin_lon, cos_lon, cos_lat * sin_lon],
        [cos_lat, 0, sin_lat]
    ])

    d_local = np.array([NS, EW, R])

    d_ecef = T @ d_local

    return d_ecef[0], d_ecef[1], d_ecef[2]

def apply_displacements(df):
    """
    Converts displacements from local topocentric to ECEF and adds them to the coordinates.

    Parameters:
    df (pd.DataFrame): DataFrame containing the original coordinates and displacements.

    Returns:
    pd.DataFrame: DataFrame with updated coordinates.
    """
    df[['latitude_rad', 'longitude_rad']] = df.apply(lambda row: ecef_to_geodetic(
        row['STAX'], row['STAY'], row['STAZ']
    ), axis=1, result_type='expand')

    df[['dX', 'dY', 'dZ']] = df.apply(lambda row: topo_to_ecef(
        row['latitude_rad'], row['longitude_rad'], row['NS'], row['EW'], row['R']
    ), axis=1, result_type='expand')

    df['STAX'] += df['dX']
    df['STAY'] += df['dY']
    df['STAZ'] += df['dZ']

    return df

import pandas as pd
import re
from io import StringIO

def write_sinex_versatile(sinex_path_in, id_block, df_update, sinex_path_out=None, suffix_name=""):
    """
    Update a block in a SINEX file with the data from a DataFrame and save the new file.

    Parameters:
    sinex_path_in (str): Path to the input SINEX file.
    id_block (str): Name of the block to be updated (without "+" or "-").
    df_update (pd.DataFrame): DataFrame containing the new data to replace the existing block.
    sinex_path_out (str, optional): Path to save the updated SINEX file. If not provided, the file will be saved with a prefix "UPD_" in the same location.
    suffix_name (str): Optional suffix for the output SINEX file.

    Returns:
    None
    """
    id_block_strt = r"\+" + id_block
    id_block_end = r"\-" + id_block

    with open(sinex_path_in, 'r') as file:
        content = file.read()

    block_start_match = re.search(id_block_strt, content)
    block_end_match = re.search(id_block_end, content)

    if not block_start_match or not block_end_match:
        raise ValueError(f"Block {id_block} not found in the SINEX file.")

    content_before = content[:block_start_match.start()]
    content_after = content[block_end_match.end():]

    header = df_update.to_string(index=False, header=True)
    block_content = f"{id_block_strt[1:]}\n{header}\n{id_block_end[1:]}\n"
    block_content = block_content.replace(' INDEX','*INDEX')

    updated_content = content_before + block_content + content_after

    if not sinex_path_out:
        sinex_path_out = sinex_path_in + suffix_name

    with open(sinex_path_out, 'w') as file:
        file.write(updated_content)

def to_scientific_notation_snx(num, digits):
    """
    Converts a number to SINEX-compatible scientific notation.

    Parameters:
    num (float): Number to convert.
    digits (int): Number of digits after the decimal point.

    Returns:
    str: Number in SINEX-compatible scientific notation.
    """
    scientific_str = "{:.{}e}".format(num, digits)
    coefficient, exponent = scientific_str.split('e')

    coefficient = float(coefficient) / 10
    exponent = int(exponent) + 1

    coefficient_str = f"{abs(coefficient):.{digits - 2}f}"

    if coefficient < 0:
        return f"-.{coefficient_str[2:]}E+{exponent:02d}"
    else:
        return f"0.{coefficient_str[2:]}E+{exponent:02d}"

def to_scientific_notation_snx_dev(num, digits):
    """
    Converts a number to SINEX-compatible scientific notation for standard deviations.

    Parameters:
    num (float): Number to convert.
    digits (int): Number of digits after the decimal point.

    Returns:
    str: Number in SINEX-compatible scientific notation for standard deviations.
    """
    scientific_str = "{:.{}e}".format(num, digits)
    coefficient, exponent = scientific_str.split('e')

    coefficient = float(coefficient) / 10
    exponent = int(exponent) + 1

    coefficient_str = f"{abs(coefficient):.{digits - 1}f}"

    return f".{coefficient_str[2:]}E{exponent:03d}"

def read_sinex_versatile(sinex_path_in, id_block, convert_date_2_dt=True, header_line_idx=-1,
                         improved_header_detection=True, verbose=False):
    """
    Reads a block from a SINEX file and returns the data as a DataFrame. @ FROME GEODEZYX TOOLBOX

    Parameters:
    sinex_path_in (str): Path to the input SINEX file.
    id_block (str): Name of the block to be read (without "+" or "-").
    convert_date_2_dt (bool): Whether to convert SINEX formatted dates to datetime.
    header_line_idx (int): Line index for the block header, default is the last line.
    improved_header_detection (bool): Whether to use improved header detection.
    verbose (bool): Whether to print header and field size information.

    Returns:
    pd.DataFrame: DataFrame containing the block data.
    """
    if id_block in ("+", "-"):
        id_block = id_block[1:]

    id_block_strt = "\+" + id_block
    id_block_end = "\-" + id_block

    Lines_list = utils.extract_text_between_elements_2(sinex_path_in, id_block_strt, id_block_end)
    Lines_list = Lines_list[1:-1]

    if not Lines_list:
        print("ERR : read_sinex_versatile : no block found, ", id_block)

    Lines_list_header = []
    Lines_list_OK = []
    header_lines = True
    for i_l, l in enumerate(Lines_list):
        if not l[0] in (" ", "\n") and header_lines:
            Lines_list_header.append(l)
        elif l[0] in (" ", "\n"):
            header_lines = False
            Lines_list_OK.append(l)
        else:
            continue

    if len(Lines_list_header) > 0 and header_line_idx:
        header_line = Lines_list_header[header_line_idx]
        header_line = header_line.replace(' VALUE', '_VALUE')
        Header_split = header_line.split()
        if not improved_header_detection:
            Fields_size = [len(e) + 1 for e in Header_split]
        else:
            Fields_size = []
            for fld_head_split in Header_split:
                if fld_head_split[0] == "*":
                    fld_head_regex = re.compile("\*" + fld_head_split[1:] + " *")
                else:
                    fld_head_regex_str = fld_head_split + " *"
                    fld_head_regex_str = fld_head_regex_str.replace("[", "\[")
                    fld_head_regex_str = fld_head_regex_str.replace("]", "\]")
                    fld_head_regex_str = fld_head_regex_str.replace("(", "\(")
                    fld_head_regex_str = fld_head_regex_str.replace(")", "\)")

                    fld_head_regex = re.compile(fld_head_regex_str)

                fld_head_space = fld_head_regex.search(header_line)

                Fields_size.append(len(fld_head_space.group()))

        if verbose:
            print("INFO : read_sinex_versatile : Auto detected column names/sizes")
            print("**** Raw header line in the file:")
            print(header_line)
            print("**** Splited header for the DataFrame:")
            print(Header_split)
            print("**** Size of the fields")
            print(Fields_size)

        Lines_str_w_head = header_line + "".join(Lines_list_OK)
        Fields_size[-1] = Fields_size[-1] + 10
        try:
            DF = pd.read_fwf(StringIO(Lines_str_w_head), widths=Fields_size)
        except pd.errors.EmptyDataError as ee:
            print("ERR: something goes wrong in the header index position")
            print("     try to give its right position manually with header_line_idx")
            raise (ee)

        DF = DF.set_axis(Header_split, axis=1)

        DF.rename(columns={DF.columns[0]: DF.columns[0][1:]}, inplace=True)

    else:
        Lines_str = "".join(Lines_list_OK)
        DF = pd.read_csv(StringIO(Lines_str), header=None, delim_whitespace=True)

    regex_time = "(([0-9]{2}|[0-9]{4}):[0-9]{3}|[0-9]{7}):[0-9]{5}"
    for col in DF.columns:
        if convert_date_2_dt and re.match(regex_time, str(DF[col].iloc[0])):
            try:
                DF[col] = DF[col].apply(lambda x: conv.datestr_sinex_2_dt(x))
            except Exception as e:
                print("WARN : read_sinex_versatile : convert date string to datetime failed")
                print(e)
                pass

    return DF
