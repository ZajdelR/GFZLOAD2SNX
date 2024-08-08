import gfzload2pkl_tools as gfzpkltl
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Process and combine station data.")
    parser.add_argument("station", type=str, help="Station name.")
    parser.add_argument("--frame", type=str, default='cf', help="Coordinate frame to use (default: 'cf').")
    parser.add_argument("--model", type=str, default='AOSH', help="Model name to use (default: 'AOSH').")
    parser.add_argument("--remove_low_freqs", type=int, default=500,
                        help="Cutoff frequency in days for filtering (default: 500).")
    parser.add_argument("--plot_results", action='store_true', help="Plot the results if specified.")
    parser.add_argument("--inp_file_dir", type=str, default='SAMPLE_NTLD_GFZ',
                        help="Input file directory (default: 'SAMPLE_NTLD_GFZ').")
    parser.add_argument("--out_dir", type=str, default='SOLUTION_PICKLES',
                        help="Output directory (default: 'SOLUTION_PICKLES').")

    # For testing purposes, you can override sys.argv like this:
    # sys.argv = ["script_name", "ABMF", "--frame=cf", "--model=AOSH", "--remove_low_freqs=500", "--plot_results", "--inp_file_dir=SAMPLE_NTLD_GFZ", "--out_dir=SOLUTION_PICKLES"]

    args = parser.parse_args()

    # Load files
    dataframes = gfzpkltl.load_station_data(args.station, args.frame, args.model, args.inp_file_dir)

    # Filter frequencies
    filtered_dataframes = gfzpkltl.filter_data(dataframes, args.remove_low_freqs)

    # Combine selected files and save
    combined_df = gfzpkltl.combine_and_save_data(filtered_dataframes, args.station, args.model, args.frame,
                                                 args.out_dir)

    # Plot the results if needed
    if args.plot_results:
        gfzpkltl.plot_results(filtered_dataframes, combined_df, args.station, args.model)

if __name__ == "__main__":
    main()