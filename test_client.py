import requests
import numpy as np

# Load EEG sample data
eeg_data = np.load("X.npy", allow_pickle=True)[7]  # Update index if needed

# Send request to backend
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"eeg_data": eeg_data.tolist()}
)

# Handle response
if response.status_code == 200:
    result = response.json()
    print("Prediction:", result["prediction"])
    print("Plot URL:", f"http://127.0.0.1:5000{result['plot_url']}")
else:
    print("Error:", response.text)
