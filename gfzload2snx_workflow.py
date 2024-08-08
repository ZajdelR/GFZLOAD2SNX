import logging
import os
import gfzload2snx_tools as gfztl
from geodezyx import conv
import pandas as pd

def setup_logging():
    """
    Sets up the logging configuration with a specific format and level.
    """
    logging.basicConfig(filename='log_gfzload2snx.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_inputs(snx_path, model_id, frame, mode):
    """
    Validates the inputs provided by the user.

    Parameters:
    snx_path (str): Path to the SINEX file.
    model_id (str): Identifier of the model to use.
    frame (str): Reference frame.
    mode (str): Mode of operation, either 'correct' or 'replace'.

    Raises:
    FileNotFoundError: If the SINEX file is not found.
    ValueError: If any of the other inputs are invalid.
    """
    if not os.path.isfile(snx_path):
        logging.error(f"SINEX file not found: {snx_path}")
        raise FileNotFoundError(f"SINEX file not found: {snx_path}")
    if not isinstance(model_id, str) or not model_id:
        logging.error(f"Invalid model_id: {model_id}")
        raise ValueError(f"Invalid model_id: {model_id}")
    if not isinstance(frame, str) or not frame:
        logging.error(f"Invalid frame: {frame}")
        raise ValueError(f"Invalid frame: {frame}")
    if not isinstance(mode, str) or not frame:
        logging.error(f"Invalid mode: {mode}")
        raise ValueError(f"Invalid mode: {mode}")

def process_station(apr_crd, station, model_id, frame, mode):
    """
    Processes a station by reading model displacements and applying them to the apriori coordinates.

    Parameters:
    apr_crd (pd.DataFrame): DataFrame containing apriori coordinates.
    station (str): Station code.
    model_id (str): Identifier of the model to use.
    frame (str): Reference frame.
    mode (str): Mode of operation, either 'correct' or 'replace'.

    Returns:
    pd.DataFrame: DataFrame with updated coordinates or None if processing fails.
    """
    apr_columns = apr_crd.columns
    type_column = next((x for x in apr_columns if 'TYPE' in x), None)
    epoch_column = next((x for x in apr_columns if 'EPOCH' in x), None)
    value_column = next((x for x in apr_columns if 'APRIORI' in x), None)

    if type_column is None or epoch_column is None or value_column is None:
        logging.warning(f"Missing expected columns for station {station}. Skipping...")
        return None

    if not apr_crd[type_column].str.contains("STA").all():
        logging.info(f"PARAMETER: {station} --> SKIPPED")
        return None

    logging.info(f"STATION: {station} --> PROCESSED")
    # Read model
    model_path = f"SOLUTION_PICKLES/{station}_{''.join(sorted(model_id))}_{frame}.PKL"

    if not os.path.isfile(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = pd.read_pickle(model_path)

    #merge model displacements with station coordinates
    apr_crd_pivot = pd.pivot_table(apr_crd, values=value_column, index=epoch_column, columns=type_column)
    apr_crd_pivot.index = apr_crd_pivot.index.date
    model_disp = model.loc[apr_crd_pivot.index]
    apr_crd_merged = pd.merge(apr_crd_pivot, model_disp, left_index=True, right_index=True)
    df = gfztl.apply_displacements(apr_crd_merged)
    if mode == "replace":
        df_unstacked = df[['STAX', 'STAY', 'STAZ']].unstack().reset_index()
    elif mode =="correct":
        df_temp = df[['dX', 'dY', 'dZ']]
        df_temp.columns = ['STAX', 'STAY', 'STAZ']
        df_unstacked = df_temp.unstack().reset_index()
    df_unstacked.columns = [type_column, epoch_column, 'NEW_VALUE']
    df_unstacked = df_unstacked.sort_values(by=[epoch_column, type_column])
    df_unstacked[epoch_column] = apr_crd[epoch_column].values
    apr_crd_new = pd.merge(apr_crd.copy(), df_unstacked, on=[epoch_column, type_column])
    return apr_crd_new

def process_sinex(snx_path, model_id, frame, mode):
    """
    Processes a SINEX file by applying model displacements to the stations.

    Parameters:
    snx_path (str): Path to the SINEX file.
    model_id (str): Identifier of the model to use.
    frame (str): Reference frame.
    mode (str): Mode of operation, either 'correct' or 'replace'.

    Returns:
    None
    """
    validate_inputs(snx_path, model_id, frame, mode)
    dfapr = gfztl.read_sinex_versatile(snx_path, "SOLUTION/APRIORI")
    snx_block = []

    for station in dfapr.CODE.unique():
        apr_crd = dfapr[dfapr.CODE == station]
        processed_data = process_station(apr_crd, station, model_id, frame, mode)
        if processed_data is not None:
            snx_block.append(processed_data)

    if not snx_block:
        logging.error("No valid stations processed. Exiting...")
        return

    snx_block_df = pd.concat(snx_block, axis=0)
    suffix_name = f".{model_id}_{frame}_{mode[:3].upper()}"
    update_sinex_estimates(snx_path, snx_block_df, "SOLUTION/ESTIMATE", mode, suffix_name=suffix_name)

def update_sinex_estimates(snx_path, snx_block_df, id_block, mode, suffix_name=""):
    """
    Updates the SINEX file with new estimates based on the processed station data.

    Parameters:
    snx_path (str): Path to the SINEX file.
    snx_block_df (pd.DataFrame): DataFrame containing the updated estimates.
    id_block (str): Block ID in the SINEX file to be updated.
    mode (str): Mode of operation, either 'correct' or 'replace'.
    suffix_name (str): Optional suffix for the output SINEX file name.

    Returns:
    None
    """
    dfest = gfztl.read_sinex_versatile(snx_path, id_block)
    value_column = next((x for x in dfest.columns if 'ESTIMATE' in x), None)
    epoch_column = next((x for x in dfest.columns if 'EPOCH' in x), None)
    type_column = next((x for x in dfest.columns if 'TYPE' in x), None)
    std_dev_column = next((x for x in dfest.columns if 'DEV' in x), None)

    if value_column is None or epoch_column is None or type_column is None or std_dev_column is None:
        logging.error("Missing expected columns in SOLUTION/ESTIMATE. Exiting...")
        return

    new_est = snx_block_df[['CODE', epoch_column, type_column, 'NEW_VALUE']]
    dfest_new = pd.merge(dfest, new_est, on=['CODE', epoch_column, type_column], how='left')
    dfest_new2 = dfest_new.copy()
    if mode == "replace":
        dfest_new[value_column] = dfest_new.apply(
            lambda row: row['NEW_VALUE'] if pd.notnull(row['NEW_VALUE']) else row[value_column],
            axis=1
        )
    elif mode == "correct":
        dfest_new[value_column] = dfest_new.apply(
            lambda row: row[value_column] - row['NEW_VALUE'] if pd.notnull(row['NEW_VALUE']) else row[value_column],
            axis=1
        )
    else:
        logging.error("Wrong mode specified.")
    dfest_new.drop('NEW_VALUE', inplace=True, axis=1)
    dfest_new[epoch_column] = dfest_new[epoch_column].apply(lambda x: conv.dt_2_sinex_datestr(x))
    dfest_new[value_column] = dfest_new[value_column].apply(lambda x: gfztl.to_scientific_notation_snx(x, digits=17))
    dfest_new[std_dev_column] = dfest_new[std_dev_column].apply(lambda x: gfztl.to_scientific_notation_snx_dev(x, digits=7))

    gfztl.write_sinex_versatile(snx_path, 'SOLUTION/ESTIMATE', dfest_new, suffix_name=suffix_name)
    logging.info(f"Updated SINEX file saved: {snx_path}")
