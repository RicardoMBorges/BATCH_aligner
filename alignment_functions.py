import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from pyteomics import mgf
from itertools import cycle
import plotly.graph_objects as go

def parse_mgf_files(directory_path):
    """
    Parses MGF files in the specified directory and returns a dictionary of spectra.
    
    Parameters:
    - directory_path: str, path to the directory containing the .mgf files.
    
    Returns:
    - spectra: dict, keys are batch names and values are lists of spectra.
    """
    mgf_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.mgf')]
    spectra = {}
    for mgf_file in mgf_files:
        print(f"Processing MGF file: {mgf_file}")
        batch_name = os.path.basename(mgf_file).replace('.mgf', '')
        spectra[batch_name] = []
        with open(mgf_file, 'r') as file:
            current_spectrum = None
            for line in file:
                line = line.strip()
                if line == "BEGIN IONS":
                    current_spectrum = {'params': {}, 'm/z array': [], 'intensity array': []}
                elif line == "END IONS":
                    if current_spectrum:
                        spectra[batch_name].append(current_spectrum)
                elif '=' in line:
                    key, value = line.split('=', 1)
                    current_spectrum['params'][key] = value
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        mz, intensity = map(float, parts)
                        current_spectrum['m/z array'].append(mz)
                        current_spectrum['intensity array'].append(intensity)
    return spectra

def extract_values_and_create_dfs(file_paths):
    """
    Reads _MetaboAnalyst.csv files and creates DataFrames for each file, extracting 
    scan_number, mz_value, and rt_value.
    
    Parameters:
    - file_paths: list of str, paths to the _MetaboAnalyst.csv files.
    
    Returns:
    - dfs: dict, keys are dynamic names for DataFrames, values are DataFrames.
    - first_rows: dict, keys are filenames, values are the first row of each DataFrame.
    """
    dfs = {}
    first_rows = {}

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        
        filename = os.path.basename(file_path).replace('_MetaboAnalyst.csv', '')
        df[['scan_number', 'mz_value', 'rt_value']] = df.iloc[:, 0].str.extract(r'(\d+)/([\d.]+)mz/([\d.]+)min')
        df = df.fillna(0)
        df['scan_number'] = df['scan_number'].astype(int)
        df['mz_value'] = df['mz_value'].astype(float)
        df['rt_value'] = df['rt_value'].astype(float)
        df['batch'] = filename
        df['feature_batch'] = df['scan_number'].astype(str) + "_" + filename
        
        cols = ['scan_number', 'mz_value', 'rt_value', 'batch', 'feature_batch'] + [col for col in df.columns if col not in ['scan_number', 'mz_value', 'rt_value', 'batch', 'feature_batch']]
        df = df[cols]
        
        first_rows[f'{filename}'] = df.iloc[0]
        df = df.iloc[1:]
        dfs[f'df_{filename}'] = df
    
    return dfs, first_rows

def align_features(dfs, mz_threshold=0.01, rt_threshold=0.2):
    """
    Aligns features across batches based on mz_value and rt_value.
    
    Parameters:
    - dfs: dict, DataFrames with features for each batch.
    - mz_threshold: float, threshold for aligning mz_value.
    - rt_threshold: float, threshold for aligning rt_value.
    
    Returns:
    - aligned_df: DataFrame, aligned features across batches.
    """
    aligned_features = []
    first_rows = []
    seen_features = set()
    all_features = []
    for batch_name, df in dfs.items():
        first_row = df.iloc[0]
        first_rows.append({
            'batch_name': batch_name,
            'scan_number': first_row['scan_number'],
            'mz_value': first_row['mz_value'],
            'rt_value': first_row['rt_value'],
            'intensity': first_row.iloc[4:].to_dict(),
        })
        for _, row in df.iloc[1:].iterrows():
            all_features.append({
                'batch_name': batch_name,
                'scan_number': row['scan_number'],
                'mz_value': row['mz_value'],
                'rt_value': row['rt_value'],
                'intensity': row.iloc[4:].to_dict(),
                'feature_batch': row['feature_batch']
            })

    for i, feature in enumerate(all_features):
        feature_key = (feature['scan_number'], feature['mz_value'], feature['rt_value'], feature['feature_batch'])
        if feature_key in seen_features:
            continue

        aligned = {
            'scan_number': feature['scan_number'],
            'mz_value': feature['mz_value'],
            'rt_value': feature['rt_value'],
            'feature_batch': feature['feature_batch'],
            'intensities': {feature['batch_name']: feature['intensity']},
            'aligned_features': [feature['feature_batch']]
        }

        for j, other_feature in enumerate(all_features):
            if i != j:
                mz_diff = abs(feature['mz_value'] - other_feature['mz_value'])
                rt_diff = abs(feature['rt_value'] - other_feature['rt_value'])

                if mz_diff <= mz_threshold and rt_diff <= rt_threshold:
                    aligned['intensities'][other_feature['batch_name']] = other_feature['intensity']
                    aligned['aligned_features'].append(other_feature['feature_batch'])
                    seen_features.add((other_feature['scan_number'], other_feature['mz_value'], other_feature['rt_value'], other_feature['feature_batch']))

        aligned_features.append(aligned)
        seen_features.add(feature_key)

    flattened_features = []
    for feature in aligned_features:
        flattened_feature = {
            'scan_number': feature['scan_number'],
            'mz_value': feature['mz_value'],
            'rt_value': feature['rt_value'],
            'feature_batch': feature['feature_batch'],
            'aligned_features': '; '.join(feature['aligned_features'])
        }
        for batch, intensity_dict in feature['intensities'].items():
            for sample, intensity in intensity_dict.items():
                flattened_feature[f'{batch}_{sample}'] = intensity
        flattened_features.append(flattened_feature)

    aligned_df = pd.DataFrame(flattened_features)

    for first_row in first_rows:
        first_row_data = {
            'scan_number': first_row['scan_number'],
            'mz_value': first_row['mz_value'],
            'rt_value': first_row['rt_value'],
            'feature_batch': first_row['batch_name'],
            'aligned_features': first_row['batch_name']
        }
        for sample, intensity in first_row['intensity'].items():
            first_row_data[f'{first_row["batch_name"]}_{sample}'] = intensity
        aligned_df = pd.concat([aligned_df, pd.DataFrame([first_row_data])], ignore_index=True)

    return aligned_df


def calculate_cosine_similarity(spectrum1, spectrum2):
    """
    Calculates the cosine similarity between two spectra.
    
    Parameters:
    - spectrum1: dict, first spectrum with 'm/z array' and 'intensity array'.
    - spectrum2: dict, second spectrum with 'm/z array' and 'intensity array'.
    
    Returns:
    - cos_sim: float, cosine similarity between the two spectra.
    """
    mz1, int1 = spectrum1['m/z array'], spectrum1['intensity array']
    mz2, int2 = spectrum2['m/z array'], spectrum2['intensity array']
    
    mz_common = np.union1d(mz1, mz2)
    int1_common = np.interp(mz_common, mz1, int1, left=0, right=0)
    int2_common = np.interp(mz_common, mz2, int2, left=0, right=0)
    
    cos_sim = cosine_similarity([int1_common], [int2_common])[0][0]
    return cos_sim

def filter_aligned_features(aligned_df, spectra, project_prefix, cosine_threshold=0.9):
    """
    Filters aligned features based on MS spectra similarity.
    
    Parameters:
    - aligned_df: DataFrame, aligned features across batches.
    - spectra: dict, keys are batch names and values are lists of spectra.
    - project_prefix: str, prefix for the project.
    - cosine_threshold: float, threshold for cosine similarity.
    
    Returns:
    - filtered_df: DataFrame, filtered aligned features based on cosine similarity.
    """
    filtered_features = []

    for _, row in aligned_df.iterrows():
        aligned_batches = row['aligned_features'].split('; ')
        if len(aligned_batches) < 2:
            continue

        similarities = []
        for i in range(len(aligned_batches) - 1):
            scan1, batch1 = aligned_batches[i].rsplit('_', 1)
            scan2, batch2 = aligned_batches[i + 1].rsplit('_', 1)
            batch1_full = project_prefix + batch1
            batch2_full = project_prefix + batch2

            spectrum1 = next((s for s in spectra[batch1_full] if s['params'].get('FEATURE_ID') == scan1), None)
            spectrum2 = next((s for s in spectra[batch2_full] if s['params'].get('FEATURE_ID') == scan2), None)

            if spectrum1 and spectrum2:
                cos_sim = calculate_cosine_similarity(spectrum1, spectrum2)
                similarities.append(cos_sim)

        if all(sim >= cosine_threshold for sim in similarities):
            filtered_features.append(row)

    filtered_df = pd.DataFrame(filtered_features)
    return filtered_df


def plot_ms_spectra(df, feature_batch, spectra, project_prefix):
    """
    Plots MS spectra for a selected feature using Plotly with vertical lines.
    
    Parameters:
    - df: DataFrame, contains aligned features.
    - feature_batch: str, the feature batch identifier.
    - spectra: dict, keys are batch names and values are lists of spectra.
    - project_prefix: str, prefix for the project.
    """
    try:
        row = df[df['feature_batch'] == feature_batch].iloc[0]
        aligned_features = row['aligned_features'].split('; ')
        
        fig = go.Figure()
        colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])

        for batch in aligned_features:
            scan_with_prefix, batch_name = batch.rsplit('_', 1)
            scan = scan_with_prefix.split('_')[0]
            full_batch_name = project_prefix + batch_name

            spectrum = next((s for s in spectra[full_batch_name] if s['params'].get('FEATURE_ID') == scan), None)

            if spectrum:
                mz_values = spectrum['m/z array']
                intensity_values = spectrum['intensity array']
                label = f'{batch_name} (scan {scan})'.replace('_', ' ')
                color = next(colors)
                for mz, intensity in zip(mz_values, intensity_values):
                    fig.add_trace(go.Scatter(x=[mz, mz], y=[0, intensity], mode='lines', line=dict(color=color), name=label, showlegend=False))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=color), name=label))

        fig.update_layout(title=f'MS Spectra for Aligned Feature: {feature_batch}',
                          xaxis_title='m/z',
                          yaxis_title='Intensity',
                          legend_title='Spectra')
        fig.show()
    except IndexError:
        print(f"No data found for feature batch: {feature_batch}")

def plot_random_ms_spectra(df, spectra, project_prefix):
    """
    Plots a random MS spectrum using Plotly with vertical lines.
    
    Parameters:
    - df: DataFrame, contains aligned features.
    - spectra: dict, keys are batch names and values are lists of spectra.
    - project_prefix: str, prefix for the project.
    """
    random_row = df.sample(n=1).iloc[0]
    feature_batch = random_row['feature_batch']
    plot_ms_spectra(df, feature_batch, spectra, project_prefix)
