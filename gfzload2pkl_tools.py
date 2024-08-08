import glob
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import scipy.signal as signal

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
        df = pd.read_csv(file_path, sep="\s+", comment='#', header=None)
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

def load_station_data(station, frame, model, inp_file_dir):
    """Load the station data from the specified directory."""
    print(f'Processing station {station}')
    station_paths = glob.glob(os.path.join(inp_file_dir, f'{station}*.{frame}'))
    station_paths = [x for x in station_paths if any(model_file in x for model_file in get_model_files(model))]
    return load_files(station_paths)

def filter_data(dataframes, remove_low_freqs):
    """Filter the dataframes based on the low frequency cutoff."""
    if remove_low_freqs:
        return filter_frequencies(dataframes, cutoff_days=remove_low_freqs)
    else:
        return dataframes

def combine_and_save_data(filtered_dataframes, station, solution_name, frame, out_dir):
    """Combine the filtered dataframes and save them to a file."""
    combined_df, solution_name = combine_selected_files(filtered_dataframes)
    output_file = f'{station}_{solution_name}_{frame}.PKL'
    output_file = os.path.join(out_dir, output_file)
    save_combined_data(combined_df, output_file)
    print(f"{station} saved successfully to {output_file}")
    return combined_df

def plot_results(filtered_dataframes, combined_df, station, solution_name):
    n_cols = len(filtered_dataframes[next(iter(filtered_dataframes))].columns)
    n_rows = len(filtered_dataframes) + 1  # Plus one for the combined DataFrame

    # Create subplots with n_cols columns and n_rows rows, sharing only the x-axis
    f, a = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=True)

    # Loop over the filtered DataFrames and plot them in the corresponding subplot
    for row, (name, df) in enumerate(filtered_dataframes.items()):
        for col, column in enumerate(df.columns):
            df[column].plot(ax=a[row, col])
            if row == 0:  # Add column titles only on the first row
                a[row, col].set_title(column)
            if col == 0:  # Add series name as y-label only for the first column
                a[row, col].set_ylabel(name)

    # Plot the combined DataFrame on the last row
    for col, column in enumerate(combined_df.columns):
        combined_df[column].plot(ax=a[-1, col])
        if col == 0:  # Add combined series name as y-label only for the first column
            a[-1, col].set_ylabel(f"{station}_{solution_name}")

    # Set specific y-limits for each column
    for row in range(n_rows):
        a[row, 0].set_ylim(-0.01, 0.01)   # First column
        a[row, 1].set_ylim(-0.002, 0.002) # Second column
        a[row, 2].set_ylim(-0.002, 0.002) # Third column

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


