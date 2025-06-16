import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Mount Google Drive (if using Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
file_path = "/content/drive/My Drive/weather_classification_data.csv"  # Change this to your actual dataset path
df = pd.read_csv(file_path)

# Convert 'Weather Type' to string format to ensure proper encoding
df['Weather Type'] = df['Weather Type'].astype(str)

# Encode the weather labels
label_encoder = LabelEncoder()
df['Weather Type'] = label_encoder.fit_transform(df['Weather Type'])

# Save label encoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Print label encoding mapping
print("Weather Labels Mapping:", dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)))

# Drop missing values
df = df.dropna()

# Split dataset into features and target variable
X = df.drop(columns=['Weather Type'])  # Features
y = df['Weather Type']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Function to predict weather
def predict_weather(temp, humidity, wind_speed, pressure, uv_index, visibility, precipitation):
    # Ensure input data has the correct column names
    input_data = np.array([[temp, humidity, wind_speed, precipitation, pressure, uv_index, visibility]])
    
    feature_names = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Apply scaling
    input_scaled = scaler.transform(input_df)
    
    # Predict
    encoded_prediction = model.predict(input_scaled)
    
    # Convert numeric prediction back to weather name
    predicted_weather = label_encoder.inverse_transform([encoded_prediction[0]])
    
    return predicted_weather[0]
  
import pickle

# Save the trained model
with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
