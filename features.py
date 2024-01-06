import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to create segments from data
def create_segments(data, window_size):
    segments = []
    for i in range(0, len(data), window_size):
        segment = data.iloc[i:i+window_size]
        if len(segment) == window_size:
            segments.append(segment)
    return segments

# Function to calculate statistical features for each EEG frequency bands
def calculate_statistical_features(segment):
    features = {}
    for frequency_band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
        electrodes = [f'{frequency_band}_{electrode}' for electrode in ['TP9', 'AF7', 'AF8', 'TP10']]
        frequency_data = segment[electrodes]
        features[f'{frequency_band}_mean'] = frequency_data.mean().values
        features[f'{frequency_band}_median'] = frequency_data.median().values
        features[f'{frequency_band}_std'] = frequency_data.std().values
        features[f'{frequency_band}_min'] = frequency_data.min().values
        features[f'{frequency_band}_max'] = frequency_data.max().values
    return features

# Function to calculate temporal trends using linear regression
def calculate_temporal_trends(segment):
    features = {}
    numeric_columns = segment.select_dtypes(include=np.number).columns
    segment.loc[:, numeric_columns] = segment[numeric_columns].fillna(segment[numeric_columns].mean())
    time_values = (segment.index - segment.index[0]).total_seconds().values.reshape(-1, 1)
    for column in segment.columns:
        model = LinearRegression()
        if column not in numeric_columns or segment[column].isnull().any():
            continue
        model.fit(time_values, segment[column].values)
        features[f'{column}_trend'] = model.coef_[0]
    return features

# Function to calculate correlation between electrodes and frequency bands
def calculate_correlation(segment):
    features = {}
    numeric_columns = segment.select_dtypes(include=np.number).columns
    correlation_matrix = segment[numeric_columns].corr()
    for electrode in ['TP9', 'AF7', 'AF8', 'TP10']:
        for frequency_band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
            column = f'{frequency_band}_{electrode}'
            if column not in numeric_columns:
                continue
            features[f'{column}_correlation'] = correlation_matrix[column].sum() - 1
    return features

# List of file paths for different datasets
file_paths = [
    'datasets/concentration_2021-07-07--12-32-23_4031631162162162619.csv',
    'datasets/med_2021-07-07--12-36-23_4031631162162162619.csv',
    'datasets/Meditative_2021-07-06--17-28-41_1605648304345181728.csv',
    'datasets/neu_2021-07-07--13-08-27_4576604433855122072.csv',
    'datasets/Neutral_2021-07-06--17-26-04_3567260142020984011.csv',
    'datasets/neutral_2021-07-07--12-34-23_1343048316368309635.csv'
]

# Process each dataset
for file_path in file_paths:
    # Read the data
    data = pd.read_csv(file_path)
    data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], format='%M:%S.%f')
    data = data.sort_values(by='TimeStamp')
    data = data.set_index('TimeStamp')

    # Window size in milliseconds (2 seconds in this case)
    window_size = 2000
    segments = create_segments(data, window_size)

    # Create a new DataFrame to store the features
    features_df = pd.DataFrame()

    # Iterate over each segment and calculate additional features
    for segment_idx, segment in enumerate(segments):
        segment_features = calculate_statistical_features(segment)
        segment_features.update(calculate_temporal_trends(segment))
        segment_features.update(calculate_correlation(segment))
        features_df = pd.concat([features_df, pd.DataFrame(segment_features)], ignore_index=True)

    # Save the new DataFrame to a CSV file
    output_file_path = f'features_output_{file_path.split("/")[-1].replace(".csv", "")}.csv'
    features_df.to_csv(output_file_path, index=False)

    # Display the features DataFrame
    print(f"Features for {file_path} saved to {output_file_path}")