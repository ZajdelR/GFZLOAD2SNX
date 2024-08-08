# GFZLOAD2SNX

GFZLOAD2SNX is a collection of Python tools designed to download, process, and analyze SINEX (Solution INdependent EXchange format) files and related geodetic data from the GFZ (German Research Centre for Geosciences). These tools facilitate the application of GFZ model displacements to SINEX files and support various models and processing workflows.

## Table of Contents

- [Introduction](#introduction)
- [Scripts Overview](#scripts-overview)
  - [download_snx.py](#download_snxpy)
  - [gfzload2pkl.py](#gfzload2pklpy)
  - [gfzload2snx.py](#gfzload2snxpy)
- [Installation](#installation)
- [Usage](#usage)
  - [Downloading SINEX Files](#downloading-sinex-files)
  - [Processing GFZ Data to Pickle Files](#processing-gfz-data-to-pickle-files)
  - [Applying GFZ Model Displacements to SINEX Files](#applying-gfz-model-displacements-to-sinex-files)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

The GFZLOAD2SNX project is designed to streamline the processing and analysis of geodetic data, specifically focusing on SINEX files. It provides tools for downloading SINEX files, processing GFZ loading dataset, converting data to pickle format, and applying GFZ model displacements to SNX files. The project supports different models and coordinate frames, ensuring flexibility and precision in data processing.

## Scripts Overview

### download_snx.py

This script handles the downloading and unzipping of SINEX files from the CDDIS FTP server. It supports filtering by GPS week and analysis center code.

**Usage:**
```
python download_snx.py --start_week <START_WEEK> --end_week <END_WEEK> --local_directory <LOCAL_DIR> --ac <ANALYSIS_CENTER> --cddis_dir <CDDIS_DIRECTORY>
```
### gfzload2pkl.py

This script processes GFZ loading file for given station and converts it into pickle format for further analysis. It allows for filtering based on frequency and supports plotting the results.

**Usage:**
```
python gfzload2pkl.py <STATION> --frame <COORDINATE_FRAME> --model <MODEL_NAME> --remove_low_freqs <CUTOFF_FREQUENCY> --plot_results --inp_file_dir <INPUT_DIRECTORY> --out_dir <OUTPUT_DIRECTORY>
```
### gfzload2snx.py

This script processes SINEX files by applying GFZ model displacements. It allows for selecting the model ID, reference frame, and processing mode (correct or replace).

**Usage:**
```
python gfzload2snx.py <SNX_PATH> <MODEL_ID> <FRAME> <MODE>
```
## Installation

1. Clone the repository:
   
   git clone https://github.com/yourusername/GFZLOAD2SNX.git

2. Navigate to the project directory:
   
   cd GFZLOAD2SNX

3. Install the required dependencies using Poetry:
   
   poetry install

## Acknowledgements

We would like to thank the German Research Centre for Geosciences (GFZ) for providing the data and models used in this project. Additionally, we acknowledge the contributions of the open-source community for the tools and libraries that made this project possible.

The repository developed under the "BENEFITS OF INTEGRATED GNSS 4 GEODESY, GEOPHYSICS, AND GEODYNAMICS" project, was facilitated through the MERIT Postdoctoral Fellowship. This project received funding from the European Commission, specifically from the MSCA-COFUND Horizon Europe call, along with additional financial support from the Central Bohemian Region's budget.

![MERIT_FOOTER.png](./img/MERIT_FOOTER.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## Contact

If you have any questions or need further assistance, feel free to contact me: radoslaw.zajdel@pecny.cz
