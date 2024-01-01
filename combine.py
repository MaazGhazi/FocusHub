import pandas as pd
from sklearn.utils import shuffle

# List of dataset files
dataset_info = [
    'concentration_2021-07-06--17-35-50_1168978808786958657.csv',
    'Meditative_2021-07-06--17-28-41_1605648304345181728.csv',
    'med_2021-07-07--12-36-23_4031631162162162619.csv',
    'Neutral_2021-07-06--17-26-04_3567260142020984011.csv',
    'neu_2021-07-07--13-08-27_4576604433855122072.csv',
    'neutral_2021-07-07--12-34-23_1343048316368309635.csv',
]

# Combine datasets into one DataFrame
combined_df = pd.concat([pd.read_csv(file) for file in dataset_info], ignore_index=True)

# Shuffle the rows
combined_df_shuffled = shuffle(combined_df, random_state=42)

# Save the shuffled DataFrame to a new CSV file
combined_df_shuffled.to_csv('combined_dataset.csv', index=False)
