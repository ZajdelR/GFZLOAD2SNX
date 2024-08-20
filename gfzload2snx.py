import sys
import traceback
import gfzload2snx_workflow as gfzwkf
import argparse
import logging

def main():
    """
    Main function to handle command-line arguments and initiate the SINEX file processing.

    This script processes a SINEX file by applying model displacements using the specified
    GFZ Model ID, reference frame, and processing mode.

    Command-line Arguments:
    - snx_path: Path to the SINEX file (str).
    - model_id: GFZ Model ID to apply (choices: ['A','AHO','AHOS','AO','AOS','H','O','S']).
    - frame: Reference frame to use (choices: ['cf', 'cm']).
    - mode: Processing mode to use (choices: ['correct', 'replace']).

    Example:
    python script_name.py SAMPLE_SNX_FILES/COD0R03FIN/COD0R03FIN_20181260000_01D_01D_SOL.SNX AHOS cf replace

    Returns:
    - None
    """
    gfzwkf.setup_logging()

    parser = argparse.ArgumentParser(description="Process SINEX file and apply model displacements.")
    parser.add_argument('snx_path', type=str, help="Path to the SINEX file")
    parser.add_argument('model_id', type=str, help="GFZ Model ID", choices=['A', 'AHO', 'AHOS', 'AO', 'AOS', 'H', 'O', 'S'])
    parser.add_argument('frame', type=str, help="Reference frame", choices=['cf', 'cm'])
    parser.add_argument('mode', type=str, help="Processing mode", choices=['correct', 'replace'])
    parser.add_argument('change_part', type=str, help="What to change", choices=['apriori', 'estimate'])

    # UNCOMMENT IF YOU WANT TO RUN IN DEBUG MODE
    #sys.argv = ['x', 'SAMPLE_SNX_FILES/COD0R03FIN/COD0R03FIN_20181260000_01D_01D_SOL.SNX', 'AHOS', 'cf', 'correct', 'apriori']
    args = parser.parse_args()

    try:
        gfzwkf.process_sinex(**vars(args))
    except Exception as e:
        logging.error("An error occurred:\n" + "".join(traceback.format_exception(None, e, e.__traceback__)))

if __name__ == "__main__":
    main()
