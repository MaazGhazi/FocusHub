import pandas as pd
from sklearn.utils import shuffle

# List of feature files to combine
feature_files = [
    'features_output_concentration_2021-07-06--17-35-50_1168978808786958657.csv',
    'features_output_med_2021-07-07--12-36-23_4031631162162162619.csv',
    'features_output_med_2021-07-07--12-36-23_4031631162162162619.csv',
    'features_output_med_2021-07-07--12-36-23_4031631162162162619.csv',
    'features_output_Neutral_2021-07-06--17-26-04_3567260142020984011.csv',
    'features_output_neutral_2021-07-07--12-34-23_1343048316368309635.csv'
]

# Create an empty DataFrame to store the combined features
combined_features = pd.DataFrame()

# Iterate over each feature file and concatenate into the combined_features DataFrame
for file in feature_files:
    features_df = pd.read_csv(file)
    
    # Drop specified columns
    features_df = features_df.drop(columns=['RAW_TP9_trend', 'RAW_AF7_trend', 'RAW_AF8_trend', 'RAW_TP10_trend',
                                            'AUX_RIGHT_trend', 'Accelerometer_X_trend', 'Accelerometer_Y_trend',
                                            'Accelerometer_Z_trend', 'Gyro_X_trend', 'Gyro_Y_trend', 'Gyro_Z_trend',
                                            'HeadBandOn_trend', 'HSI_TP9_trend', 'HSI_AF7_trend', 'HSI_AF8_trend',
                                            'HSI_TP10_trend', 'Battery_trend'])
    
    combined_features = pd.concat([combined_features, features_df], ignore_index=True)

# Shuffle the rows
combined_features = shuffle(combined_features, random_state=42)

# Move "Result_trend" column to the last position
result_trend_column = combined_features.pop('Result_trend')
combined_features['Result_trend'] = result_trend_column

# Save the combined features to a new CSV file
combined_features.to_csv('shuffled_combined_features.csv', index=False)

# Display the combined features DataFrame
print(combined_features)
