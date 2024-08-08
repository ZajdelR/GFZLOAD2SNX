import numpy as np
import scipy.signal as signal
import os
from geodezyx import utils
from geodezyx import conv

# Define a function to map the model identifier to the model type
def get_model_files(model_descriptor):
    mapper = {'A': 'ntal', 'O': 'ntol', 'S': 'slel', 'H': 'cwsl'}
    model = [mapper[x] for x in model_descriptor]
    return model

def get_model_type(filename):
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
    # List all files in the given directory
    files = os.listdir(directory)

    # Extract the first 4 characters from each file name
    prefixes = [file[:4] for file in files if os.path.isfile(os.path.join(directory, file))]

    # Get unique prefixes
    unique_prefixes = set(prefixes)

    return unique_prefixes


def load_files(file_paths):
    dataframes = {}
    for file_path in file_paths:
        # Load the file into a DataFrame, assuming space-delimited or other delimiter
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
    # Convert cutoff from days to frequency (Hz)
    # Assuming time is in days and sampling rate is 1 day
    cutoff_freq = 1 / cutoff_days
    (b, a) = signal.butter(1, cutoff_freq, btype='high', fs=1)
    filtered_data = signal.filtfilt(b, a, data[column])
    return filtered_data


def filter_frequencies(dataframes, cutoff_days=500):
    filtered_dataframes = {}
    for key, df in dataframes.items():
        filtered_df = df.copy()
        # Apply high-pass filter to specific columns
        for column in df.columns:
            if column in ['R', 'EW', 'NS']:
                filtered_df[column] = high_pass_filter(df, column, cutoff_days)
        filtered_dataframes[key] = filtered_df
    return filtered_dataframes


def combine_selected_files(filtered_dataframes):
    combined_df = sum(filtered_dataframes.values())
    solution_name = ''.join(sorted([x.split('_')[-1] for x in list(filtered_dataframes.keys())]))
    return combined_df, solution_name


def save_combined_data(combined_df, output_file):
    combined_df.to_pickle(output_file)


def ecef_to_geodetic(x, y, z):
    """
    Converts ECEF coordinates to geodetic coordinates (latitude, longitude, height).

    Parameters:
    x, y, z (float): ECEF coordinates.

    Returns:
    tuple: Latitude and longitude in radians.
    """
    # WGS-84 ellipsiod parameters
    a = 6378137.0  # Semi-major axis
    e2 = 6.69437999014e-3  # Square of eccentricity

    # Calculations
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
    # Create the transformation matrix from local topocentric to ECEF
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Transformation matrix
    T = np.array([
        [-sin_lat * cos_lon, -sin_lon, cos_lat * cos_lon],
        [-sin_lat * sin_lon, cos_lon, cos_lat * sin_lon],
        [cos_lat, 0, sin_lat]
    ])

    # Local displacements vector
    d_local = np.array([NS, EW, R])

    # Convert to ECEF displacements
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
    # Convert ECEF to geodetic coordinates to get latitude and longitude
    df[['latitude_rad', 'longitude_rad']] = df.apply(lambda row: ecef_to_geodetic(
        row['STAX'], row['STAY'], row['STAZ']
    ), axis=1, result_type='expand')

    # Apply the displacement conversion and addition
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

    Parameters
    ----------
    sinex_path_in : str
        Path to the input SINEX file.

    id_block : str
        Name of the block to be updated (without "+" or "-").

    df_update : pd.DataFrame
        DataFrame containing the new data to replace the existing block.

    sinex_path_out : str, optional
        Path to save the updated SINEX file. If not provided, the file will be saved with a prefix "UPD_" in the same location.

    header_line_idx : int or None
        If the block header contains several lines, use this line index.
        Default is -1 (the last line).

    Returns
    -------
    None
    """

    # Define start and end markers for the block
    id_block_strt = r"\+" + id_block
    id_block_end = r"\-" + id_block

    # Read the entire content of the SINEX file
    with open(sinex_path_in, 'r') as file:
        content = file.read()

    # Find the block to be replaced
    block_start_match = re.search(id_block_strt, content)
    block_end_match = re.search(id_block_end, content)

    if not block_start_match or not block_end_match:
        raise ValueError(f"Block {id_block} not found in the SINEX file.")

    # Extract the content before, within, and after the block
    content_before = content[:block_start_match.start()]
    content_after = content[block_end_match.end():]

    # Generate the new block content from the DataFrame
    header = df_update.to_string(index=False, header=True)
    block_content = f"{id_block_strt[1:]}\n{header}\n{id_block_end[1:]}\n"
    block_content = block_content.replace(' INDEX','*INDEX')
    # Combine the content
    updated_content = content_before + block_content + content_after

    # Define output file name
    if not sinex_path_out:
        sinex_path_out = sinex_path_in + suffix_name

    # Save the updated content to the new SINEX file
    with open(sinex_path_out, 'w') as file:
        file.write(updated_content)


def to_scientific_notation(num, digits):
    # Convert to scientific notation
    scientific_str = "{:.{}e}".format(num, digits)

    # Split the string into the coefficient and exponent
    coefficient, exponent = scientific_str.split('e')

    # Adjust the coefficient to match the desired format
    coefficient = float(coefficient) / 10
    exponent = int(exponent) + 1

    # Ensure the coefficient has 13 digits after the decimal point to make the final string length 15
    coefficient_str = f"{abs(coefficient):.{digits - 2}f}"

    if coefficient < 0:
        return f"-.{coefficient_str[2:]}E+{exponent:02d}"
    else:
        return f"0.{coefficient_str[2:]}E+{exponent:02d}"


def to_scientific_notation_dev(num, digits):
    # Convert to scientific notation
    scientific_str = "{:.{}e}".format(num, digits)

    # Split the string into the coefficient and exponent
    coefficient, exponent = scientific_str.split('e')

    # Adjust the coefficient to match the desired format
    coefficient = float(coefficient) / 10
    exponent = int(exponent) + 1

    # Ensure the coefficient has 13 digits after the decimal point to make the final string length 15
    coefficient_str = f"{abs(coefficient):.{digits - 1}f}"

    return f".{coefficient_str[2:]}E{exponent:03d}"


def read_sinex_versatile(sinex_path_in, id_block,
                         convert_date_2_dt=True,
                         header_line_idx=-1,
                         improved_header_detection=True,
                         verbose=True):
    """
    Read a block from a SINEX and return the data as a DataFrame

    Parameters
    ----------
    sinex_path_in : str
        Description param1

    id_block : str
        Name of the block (without "+" or "-")

    convert_date_2_dt : bool
        Try to convert a SINEX formated date as a python datetime

    header_line_idx : int or None
        If the block header contains several lines, use this line index
        Per default, the last (-1)
        For the first line, use 0
        If no header is properly defined, use None

    improved_header_detection : bool
        Improved header detection.
        Works for most cases but sometime the simple version works better.
        (advenced usage)
        Default is True

    verbose : bool
        print the header and its field size
        Default is True

    Returns
    -------
    DF : Pandas DataFrame
        Returned DataFrame
    """

    ### remove the + or - if any
    if id_block in ("+", "-"):
        id_block = id_block[1:]

    id_block_strt = "\+" + id_block
    id_block_end = "\-" + id_block

    Lines_list = utils.extract_text_between_elements_2(sinex_path_in,
                                                       id_block_strt,
                                                       id_block_end)
    Lines_list = Lines_list[1:-1]

    if not Lines_list:
        print("ERR : read_sinex_versatile : no block found, ", id_block)

    #### Remove commented lines
    Lines_list_header = []
    Lines_list_OK = []
    header_lines = True
    for i_l, l in enumerate(Lines_list):
        if not l[0] in (" ", "\n") and header_lines:
            ## here we store the 1st commented lines i.e. the header
            Lines_list_header.append(l)
        elif l[0] in (" ", "\n"):
            ## here we store the data lines (not commented)
            header_lines = False
            Lines_list_OK.append(l)
        else:
            ## here we skip the final commented lines (can happend)
            continue

    if len(Lines_list_header) > 0 and header_line_idx:
        ### define the header
        header_line = Lines_list_header[header_line_idx]
        header_line = header_line.replace(' VALUE', '_VALUE')
        Header_split = header_line.split()
        if not improved_header_detection:
            ### Simple case when the columns are splitted with only a single
            Fields_size = [len(e) + 1 for e in Header_split]
        else:
            ### Smarter case : we search for the n spaces after the column name
            Fields_size = []

            for fld_head_split in Header_split:
                if fld_head_split[0] == "*":
                    ## test to manage a * as 1st character
                    ## which can be wrongly interpreted in the regex
                    fld_head_regex = re.compile("\*" + fld_head_split[1:] + " *")
                else:
                    fld_head_regex_str = fld_head_split + " *"
                    ### update 202110: now the brackets must be escaped (normal btw...)
                    fld_head_regex_str = fld_head_regex_str.replace("[", "\[")
                    fld_head_regex_str = fld_head_regex_str.replace("]", "\]")
                    fld_head_regex_str = fld_head_regex_str.replace("(", "\(")
                    fld_head_regex_str = fld_head_regex_str.replace(")", "\)")

                    fld_head_regex = re.compile(fld_head_regex_str)

                fld_head_space = fld_head_regex.search(header_line)

                Fields_size.append(len(fld_head_space.group()))
                # print(fld_head_space.group())

                # # weak method (210216) archived for legacy
                # fld_head_regex = re.compile(fld_head_split[1:] + " *") #trick:
                # #1st char is removed, because it can be a *
                # #and then screw the regex. This char is re-added at the end
                # #when the len is stored (the "+1" below)
                # fld_head_space = fld_head_regex.search(header_line)
                # Fields_size.append(len(fld_head_space.group()) + 1)
                # ### !!!!! something is weird here !!!!!
                # print(fld_head_space.group())
                # ### and you will see a bug !!!
                # ### PS 210216

        if verbose:
            print("INFO : read_sinex_versatile : Auto detected column names/sizes")
            print("**** Raw header line in the file:")
            print(header_line)
            print("**** Splited header for the DataFrame:")
            print(Header_split)
            print("**** Size of the fields")
            print(Fields_size)

        ### Add the header in the big string
        Lines_str_w_head = header_line + "".join(Lines_list_OK)
        Fields_size[-1] = Fields_size[-1] + 10
        ### Read the file
        try:
            DF = pd.read_fwf(StringIO(Lines_str_w_head), widths=Fields_size)
        except pd.errors.EmptyDataError as ee:
            print("ERR: something goes wrong in the header index position")
            print("     try to give its right position manually with header_line_idx")
            raise (ee)

        DF = DF.set_axis(Header_split, axis=1)

        ### Rename the 1st column (remove the comment marker)
        DF.rename(columns={DF.columns[0]: DF.columns[0][1:]}, inplace=True)

    else:  # no header in the SINEX
        Lines_str = "".join(Lines_list_OK)
        DF = pd.read_csv(StringIO(Lines_str), header=None,
                         delim_whitespace=True)

    regex_time = "(([0-9]{2}|[0-9]{4}):[0-9]{3}|[0-9]{7}):[0-9]{5}"
    for col in DF.columns:
        if convert_date_2_dt and re.match(regex_time,
                                          str(DF[col].iloc[0])):
            try:
                DF[col] = DF[col].apply(lambda x: conv.datestr_sinex_2_dt(x))
            except Exception as e:
                print("WARN : read_sinex_versatile : convert date string to datetime failed")
                print(e)
                pass

    return DF
