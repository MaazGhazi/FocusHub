import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the combined data
file_path = "shuffled_combined_features.csv"
data = pd.read_csv(file_path)

# Assuming the last column is categorical (0, 1, 2)
categorical_col = data.columns[-1]

# Encode the categorical variable
label_encoder = LabelEncoder()
data[categorical_col] = label_encoder.fit_transform(data[categorical_col])

# Split the data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors (kNN)
k = 5  # You can adjust the value of k
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_scaled, y_train)
y_knn_pred = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_knn_pred)
print(f"Accuracy of K-Nearest Neighbors: {accuracy_knn}")

# Decision Tree (DT)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_dt_pred = dt_model.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test, y_dt_pred)
print(f"Accuracy of Decision Tree: {accuracy_dt}")

# Naive Bayes (NB)
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_nb_pred = nb_model.predict(X_test_scaled)
accuracy_nb = accuracy_score(y_test, y_nb_pred)
print(f"Accuracy of Naive Bayes: {accuracy_nb}")

# Linear Discriminant Analysis (LDA)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train_scaled, y_train)
y_lda_pred = lda_model.predict(X_test_scaled)
accuracy_lda = accuracy_score(y_test, y_lda_pred)
print(f"Accuracy of Linear Discriminant Analysis: {accuracy_lda}")
