import requests
import numpy as np

# Load a sample EEG data segment (same shape as model expects)
X = np.load('X.npy', allow_pickle=True)  # shape: (num_samples, channels, time)

# Pick one sample
sample_eeg = X[7]  # shape: (channels, time)

# Convert to list for JSON serialization
payload = {"eeg_data": sample_eeg.tolist()}

# Send POST request to your local Flask backend
response = requests.post("http://127.0.0.1:5000/predict", json=payload)

print("Response:", response.json())
