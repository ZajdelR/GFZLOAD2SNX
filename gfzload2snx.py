import sys
import traceback
import gfzload2snx_workflow as gfzwkf
import argparse
import logging

if __name__ == "__main__":
    gfzwkf.setup_logging()

    parser = argparse.ArgumentParser(description="Process SINEX file and apply model displacements.")
    parser.add_argument('snx_path', type=str, help="Path to the SINEX file")
    parser.add_argument('model_id', type=str, help="GFZ Model ID", choices=['A','AHO','AHOS','AO','AOS','H','O','S'])
    parser.add_argument('frame', type=str, help="Reference frame", choices=['cf','cm'])
    parser.add_argument('mode', type=str, help="Processing mode", choices=['correct', 'replace'])

    ## UNCOMMENT IF YOU WANT TO RUN IN DEBUG MODE
    sys.argv = ['disp_to_snx.py', 'SAMPLE_SNX_FILES/COD0R03FIN/COD0R03FIN_20181260000_01D_01D_SOL.SNX', 'AHOS', 'cf', 'correct']
    args = parser.parse_args()

    try:
        gfzwkf.process_sinex(**vars(args))
    except Exception as e:
        logging.error("An error occurred:\n" + "".join(traceback.format_exception(None, e, e.__traceback__)))
