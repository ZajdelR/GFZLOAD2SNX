import argparse
from ftplib import FTP_TLS
import os
import gzip
import logging
import configparser
import sys

# Setup basic configuration for logging
logging.basicConfig(filename='download_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_unzip_file(host, user, passwd, directory, filename, local_subfolder):
    """
    Downloads a file from an FTP server and unzips it locally.

    Parameters:
    - host (str): The hostname of the FTP server.
    - user (str): The username for the FTP login.
    - passwd (str): The password for the FTP login.
    - directory (str): The remote directory on the FTP server where the file is located.
    - filename (str): The name of the file to be downloaded.
    - local_subfolder (str): The local directory where the file should be saved.

    Returns:
    - None
    """
    try:
        # Establish a new secure FTP connection for each thread
        ftps = FTP_TLS(host=host)
        ftps.login(user=user, passwd=passwd)
        ftps.prot_p()  # Switch to secure data connection
        ftps.cwd(directory)

        local_filepath = os.path.join(local_subfolder, filename)
        unzip_filepath = os.path.join(local_subfolder, filename[:-3])  # Remove .gz for the new filename

        # Check if the unzipped file already exists
        if os.path.exists(unzip_filepath):
            logging.info(f"Skipping download; unzipped file {unzip_filepath} already exists.")
            ftps.quit()
            return

        # Download the file
        logging.info(f"Downloading {filename} to {local_filepath}...")
        with open(local_filepath, 'wb') as local_file:
            ftps.retrbinary(f"RETR {filename}", local_file.write)

        # Unzipping the file
        with gzip.open(local_filepath, 'rb') as f_in:
            with open(unzip_filepath, 'wb') as f_out:
                f_out.write(f_in.read())

        logging.info(f"Unzipped {filename} to {unzip_filepath}")

        # Remove the original .gz file
        os.remove(local_filepath)
        logging.info(f"Removed original file {local_filepath}")

        ftps.quit()

    except Exception as e:
        logging.error(f"Failed to process {filename} in directory {directory}: {e}")
        if 'ftps' in locals():
            ftps.quit()

def download_and_unzip_snx_files(host, user, passwd, base_directory, week, local_directory, ac):
    """
    Downloads and unzips all SNX files for a given GPS week from an FTP server.

    Parameters:
    - host (str): The hostname of the FTP server.
    - user (str): The username for the FTP login.
    - passwd (str): The password for the FTP login.
    - base_directory (str): The base directory on the FTP server where the SNX files are located.
    - week (int): The GPS week number to download files for.
    - local_directory (str): The local directory where the files should be saved.
    - ac (str): The analysis center code to filter the files.

    Returns:
    - None
    """
    directory = f"{base_directory}/{week}/"
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    ftps = FTP_TLS(host=host)  # This connection is just to list files
    ftps.login(user=user, passwd=passwd)
    ftps.prot_p()
    ftps.cwd(directory)
    files = ftps.nlst()
    ftps.quit()
    files = [filename for filename in files if filename.endswith('.SNX.gz') and filename.startswith(ac)]

    for filename in files:
        local_subfolder = os.path.join(local_directory, filename[:10])
        if not os.path.exists(local_subfolder):
            os.makedirs(local_subfolder)

        download_and_unzip_file(host, user, passwd, directory, filename, local_subfolder)

def main():
    """
    Main function to handle command-line arguments and initiate the download process.

    Command-line Arguments:
    - --start_week: Starting GPS week number (int).
    - --end_week: Ending GPS week number (int).
    - --local_directory: Local directory to save the files (str).
    - --ac: Analysis center code to filter the files (str).
    - --cddis_dir: The directory on the FTP server where the SNX files are stored (str).

    Returns:
    - None
    """
    parser = argparse.ArgumentParser(description="Download and unzip SNX files from an FTP server.")
    parser.add_argument('--start_week', type=int, required=True, help="Starting GPS week number.")
    parser.add_argument('--end_week', type=int, required=True, help="Ending GPS week number.")
    parser.add_argument('--local_directory', type=str, default="SAMPLE_SNX_FILES", help="Local directory to save the files.")
    parser.add_argument('--ac', type=str, required=True, help="Analysis center code (e.g., 'COD').")
    parser.add_argument('--cddis_dir', type=str, default='/gnss/products/repro3', help="SNX directory (e.g., '/gnss/products/repro3').")

    # For testing purposes, you can override sys.argv like this:
    sys.argv = ["script_name", "--start_week=2000", "--end_week=2000", "--ac=COD"]
    args = parser.parse_args()

    # Load credentials from an external configuration file
    config = configparser.ConfigParser()
    config.read('ftp_config.ini')
    host = config.get('FTP', 'host')
    user = config.get('FTP', 'user')
    passwd = config.get('FTP', 'passwd')

    # Loop through the range of weeks and download/unzip files for each
    for week in range(args.start_week, args.end_week + 1):
        download_and_unzip_snx_files(host, user, passwd, args.cddis_dir, week, args.local_directory, args.ac)

if __name__ == "__main__":
    main()
