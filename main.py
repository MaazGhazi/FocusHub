import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define brain wave categories
brain_wave_categories = {
    'Delta': ["Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10"],
    'Theta': ["Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10"],
    'Alpha': ["Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10"],
    'Beta': ["Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10"],
    'Gamma': ["Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10"],
}

df = pd.read_csv('combined_dataset.csv')
df.head()

for category, category_labels in brain_wave_categories.items():
    plt.figure(figsize=(12, 6))
    
    for label in category_labels:
        plt.hist(df[df["Result"]==1][label], color='blue', alpha=0.5, density=True, label="Meditative")
        plt.hist(df[df["Result"]==0][label], color='red', alpha=0.5, density=True, label="Neutral")
        plt.hist(df[df["Result"]==2][label], color='green', alpha=0.5, density=True, label="Concentration")
    
    plt.title(f"{category} Brain Wave Category Histograms")
    plt.ylabel('Probability')
    plt.xlabel('Value')
    plt.legend()
    plt.show()

train, valid, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

def scale_dataset(data_frame, oversample = False):
    X = data_frame[data_frame.columns[:-1]].values
    y = data_frame[data_frame.columns[-1]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
      ros = RandomOverSampler()
      X, y = ros.fit_resample(X,y)
      
    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y
train, X_train, y_train = scale_dataset(train, oversample= True)

print(len(train[train["Result"]==0]))
print(len(train[train["Result"]==1]))
print(len(train[train["Result"]==2]))