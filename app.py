from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
class EEGClassifier(nn.Module):
    def __init__(self, input_channels, input_time):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(input_channels, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_channels, input_time)
            x = F.relu(self.conv1(dummy_input))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            flattened_size = x.numel()

        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim (batch, 1, channels, time)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)  # flatten all but batch
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_sample = np.load('X.npy', allow_pickle=True)[0]
input_channels, input_time = X_sample.shape
model = EEGClassifier(input_channels, input_time).to(device)
model.load_state_dict(torch.load('eeg_model.pth', map_location=device))
model.eval()
app = Flask(__name__)

os.makedirs("static", exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        eeg_data = np.array(data['eeg_data'], dtype=np.float32)
        eeg_tensor = torch.from_numpy(eeg_data).unsqueeze(0)

        with torch.no_grad():
            output = model(eeg_tensor)
            prediction = torch.argmax(output, dim=1).item()

        pred_label = "left" if prediction == 0 else "right"

        # Create EEG plot
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join("static", filename)

        plt.figure(figsize=(10, 4))
        for ch in eeg_data:
            plt.plot(ch)
        plt.title(f"EEG Signal (Prediction: {pred_label})")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        return jsonify({
            "prediction": pred_label,
            "plot_url": f"/static/{filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
