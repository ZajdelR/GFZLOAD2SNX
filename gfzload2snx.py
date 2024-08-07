import argparse
import logging
from gfzload2snx_tools import *
from geodezyx import conv
import pandas as pd
import sys
import traceback

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_inputs(snx_path, model_id, frame):
    if not os.path.isfile(snx_path):
        logging.error(f"SINEX file not found: {snx_path}")
        raise FileNotFoundError(f"SINEX file not found: {snx_path}")
    if not isinstance(model_id, str) or not model_id:
        logging.error(f"Invalid model_id: {model_id}")
        raise ValueError(f"Invalid model_id: {model_id}")
    if not isinstance(frame, str) or not frame:
        logging.error(f"Invalid frame: {frame}")
        raise ValueError(f"Invalid frame: {frame}")

def process_station(apr_crd, station, model_id, frame):
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
    model_path = f"SOLUTION_PICKLES/{station}_{''.join(sorted(model_id))}_{frame}.PKL"

    if not os.path.isfile(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = pd.read_pickle(model_path)
    apr_crd_pivot = pd.pivot_table(apr_crd, values=value_column, index=epoch_column, columns=type_column)
    apr_crd_pivot.index = apr_crd_pivot.index.date
    model_disp = model.loc[apr_crd_pivot.index]
    apr_crd_merged = pd.merge(apr_crd_pivot, model_disp, left_index=True, right_index=True)
    df = apply_displacements(apr_crd_merged)
    df_ustacked = df[['STAX', 'STAY', 'STAZ']].unstack().reset_index()
    df_ustacked.columns = [type_column, epoch_column, 'NEW_VALUE']
    df_ustacked = df_ustacked.sort_values(by=[epoch_column, type_column])
    df_ustacked.loc[:, epoch_column] = apr_crd.loc[:, epoch_column].values
    apr_crd_new = pd.merge(apr_crd.copy(), df_ustacked, on=[epoch_column, type_column])
    return apr_crd_new

def process_sinex(snx_path, model_id, frame):
    validate_inputs(snx_path, model_id, frame)
    dfapr = read_sinex_versatile(snx_path, "SOLUTION/APRIORI")
    snx_block = []

    for station in dfapr.CODE.unique():
        apr_crd = dfapr[dfapr.CODE == station]
        processed_data = process_station(apr_crd, station, model_id, frame)
        if processed_data is not None:
            snx_block.append(processed_data)

    if not snx_block:
        logging.error("No valid stations processed. Exiting...")
        return

    snx_block_df = pd.concat(snx_block, axis=0)
    update_sinex_estimates(snx_path, snx_block_df, "SOLUTION/ESTIMATE", suffix_name=f".{model_id}_{frame}")

def update_sinex_estimates(snx_path, snx_block_df, id_block, suffix_name=""):
    dfest = read_sinex_versatile(snx_path, id_block)
    value_column = next((x for x in dfest.columns if 'ESTIMATE' in x), None)
    epoch_column = next((x for x in dfest.columns if 'EPOCH' in x), None)
    type_column = next((x for x in dfest.columns if 'TYPE' in x), None)
    std_dev_column = next((x for x in dfest.columns if 'DEV' in x), None)

    if value_column is None or epoch_column is None or type_column is None or std_dev_column is None:
        logging.error("Missing expected columns in SOLUTION/ESTIMATE. Exiting...")
        return

    new_est = snx_block_df[['CODE', epoch_column, type_column, 'NEW_VALUE']]
    dfest_new = pd.merge(dfest, new_est, on=['CODE', epoch_column, type_column], how='left')
    dfest_new[value_column] = dfest_new.apply(
        lambda row: row['NEW_VALUE'] if pd.notnull(row['NEW_VALUE']) else row[value_column],
        axis=1
    )
    dfest_new.drop('NEW_VALUE', inplace=True, axis=1)
    dfest_new[epoch_column] = dfest_new[epoch_column].apply(lambda x: conv.dt_2_sinex_datestr(x))
    dfest_new[value_column] = dfest_new[value_column].apply(lambda x: to_scientific_notation(x, digits=17))
    dfest_new[std_dev_column] = dfest_new[std_dev_column].apply(lambda x: to_scientific_notation_dev(x, digits=7))

    write_sinex_versatile(snx_path, 'SOLUTION/ESTIMATE', dfest_new, suffix_name=suffix_name)
    logging.info(f"Updated SINEX file saved: {snx_path}")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Process SINEX file and apply model displacements.")
    parser.add_argument('snx_path', type=str, help="Path to the SINEX file")
    parser.add_argument('model_id', type=str, help="GFZ Model ID", choices=['A','AHO','AHOS','AO','AOS','H','O','S'])
    parser.add_argument('frame', type=str, help="Reference frame", choices=['cf','cm'])
    sys.argv = ['disp_to_snx.py', 'SAMPLE_SNX_FILES/COD0R03FIN/COD0R03FIN_20181260000_01D_01D_SOL.SNX', 'AHOS', 'cf']
    args = parser.parse_args()

    try:
        process_sinex(args.snx_path, args.model_id, args.frame)
    except Exception as e:
        logging.error("An error occurred:\n" + "".join(traceback.format_exception(None, e, e.__traceback__)))
