import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


labels = [
    "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
    "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
    "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
    "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
    "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10",
    "RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10","HeadBandOn",
    "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10",
    "Battery", "Elements"
]

df = pd.read_csv('concentration_2021-07-06--17-35-50_1168978808786958657.csv')

df.head()

for label in labels:
    if label == "RAW_TP9":
        break  # Stop iterating when reaching "RAW_TP9"
    
    if label in df.columns:
        plt.figure()
        df[label].plot(kind='hist', bins=50, color='blue', edgecolor='black')
        plt.title(label + " Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()