# Batch Correction and Feature Alignment for MZmine

This repository contains scripts and functions for batch correction and feature alignment of mass spectrometry data processed with MZmine. The pipeline includes parsing MGF files, extracting values from MetaboAnalyst CSV files, aligning features across batches, filtering based on cosine similarity, and visualizing mass spectrometry spectra.

## Data Requirements:

1. Process each batch using MZMine accordingly.
2. Export the aligned feature list for applying the exported data to:
    (a) GNPS-FBMN (xxxx_quant.csv + xxxx.mgf)
    (b) MetaboAnalyst (xxxx_MetaboAnalyst.csv)
3. Name the exported files as follows: xxxx_batch#.mgf, xxxx_batch#.csv, xxxx_batch#_MetaboAnalyst.csv
    xxxx is the "project name" (e.g., "PHerb1"), and # refers to the batch number.

## Setup

### Requirements

The required Python packages are listed in the `requirements.txt` file:

- numpy
- pandas
- plotly
- matplotlib
- seaborn
- pycombat
- scikit-learn
- pyteomics

### Installation

To set up the environment and install the necessary packages, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Ensure that you have Python 3.11 installed on your system. If you don't, adjust the Python version in the `run.bat` file accordingly.

3. Run the `run.bat` file to set up a virtual environment, install the required packages, and launch Jupyter Notebook:
    ```sh
    ./run.bat
    ```

## Usage

### Functions

The core functions for processing and analyzing the data are provided in the `alignment_functions.py` file:

- **parse_mgf_files(directory_path)**: Parses MGF files in the specified directory and returns a dictionary of spectra.

- **extract_values_and_create_dfs(file_paths)**: Reads `_MetaboAnalyst.csv` files and creates DataFrames for each file, extracting `scan_number`, `mz_value`, and `rt_value`.

- **align_features(dfs, mz_threshold=0.01, rt_threshold=0.2)**: Aligns features across batches based on `mz_value` and `rt_value`.

- **calculate_cosine_similarity(spectrum1, spectrum2)**: Calculates the cosine similarity between two spectra.

- **filter_aligned_features(aligned_df, spectra, project_prefix, cosine_threshold=0.9)**: Filters aligned features based on MS spectra similarity.

- **plot_ms_spectra(df, feature_batch, spectra, project_prefix)**: Plots MS spectra for a selected feature using Plotly with vertical lines.

- **plot_random_ms_spectra(df, spectra, project_prefix)**: Plots a random MS spectrum using Plotly with vertical lines.

### Example

Below is an example of how to use the functions in a Jupyter Notebook:

```python
from alignment_functions import parse_mgf_files, extract_values_and_create_dfs, align_features, filter_aligned_features, plot_ms_spectra, plot_random_ms_spectra

# Define your directory path and project prefix
directory_path = r'C:\Users\borge\Edison_Lab@UGA Dropbox\Ricardo Borges\Projeto_Andrew\Andrew Lab\Projeto herbário\MSMS - Amostras\GNPS_Batches1'
project_prefix = "PHerb1_"

# Parse .mgf files to get spectra
spectra = parse_mgf_files(directory_path)

# Find all _MetaboAnalyst.csv files in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('_MetaboAnalyst.csv')]

# Create separate DataFrames for each file and save first rows
dfs, first_rows = extract_values_and_create_dfs(file_paths)

# Align features across batches with selected mz_threshold and rt_threshold values
mz_threshold = 0.01  # Example value for mz_threshold
rt_threshold = 0.2   # Example value for rt_threshold
aligned_rt_mz_df = align_features(dfs, mz_threshold=mz_threshold, rt_threshold=rt_threshold)

# Filter the aligned features DataFrame with selected cosine_threshold value
cosine_threshold = 0.95  # Example value for cosine_threshold
MSfiltered_aligned_features_df = filter_aligned_features(aligned_rt_mz_df, spectra, project_prefix, cosine_threshold=cosine_threshold)

# Save the filtered DataFrame
output_file_path = os.path.join(directory_path, 'alignment_info_df', 'MSfiltered_aligned_features_df.csv')
MSfiltered_aligned_features_df.to_csv(output_file_path, index=False)

# Plot MS spectra for a specific feature
feature_batch = '48_PHerb1_batch1'
plot_ms_spectra(MSfiltered_aligned_features_df, feature_batch, spectra, project_prefix)

# Plot random MS spectra
plot_random_ms_spectra(MSfiltered_aligned_features_df, spectra, project_prefix)
