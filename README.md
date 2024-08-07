# SINEX Processing Tool

## Overview

This repository provides a set of tools to process SINEX (Software INdependent EXchange format) files, particularly for geodetic applications. The main functionality includes applying model displacements to SINEX data and updating the corresponding estimates.

## Repository Structure

- **gfzload2snx.py**: Main script that orchestrates the processing of SINEX files. It validates inputs, processes stations, and updates SINEX estimates.
- **gfzload2snx_tools.py**: Contains utility functions for handling models, filtering data, and performing coordinate transformations.
- **pyproject.toml**: Project configuration file for managing dependencies and project metadata.

## Usage

### Prerequisites

Ensure you have the necessary Python packages installed. The dependencies are listed in `pyproject.toml`.

### Running the Script

To process a SINEX file, use the following command:

```python gfzload2snx.py <sinex_file_path> <model_id> <frame>
```

- `<sinex_file_path>`: Path to the SINEX file to be processed.
- `<model_id>`: Model ID to be used (e.g.,`A`, `O`, `S`, `H`).
- `<frame>`: Reference frame (`cf` or `cm`).

### Example

```python gfzload2snx.py SAMPLE_SNX_FILES/COD0R03FIN/COD0R03FIN_20181260000_01D_01D_SOL.SNX AHOS cf```

### Logging

The script provides detailed logging for debugging and monitoring purposes. Log messages are displayed in the console with timestamps and severity levels.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## Contact

If you have any questions or need further assistance, feel free to contact me: radoslaw.zajdel@pecny.cz

## Acknowledgement
The repository developed under the "BENEFITS OF INTEGRATED GNSS 4 GEODESY, GEOPHYSICS, AND GEODYNAMICS" project, was facilitated through the MERIT Postdoctoral Fellowship. This project received funding from the European Commission, specifically from the MSCA-COFUND Horizon Europe call, along with additional financial support from the Central Bohemian Region's budget.
