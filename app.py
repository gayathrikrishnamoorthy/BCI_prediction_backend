from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'eeg_data' not in data:
            return jsonify({'error': 'Missing eeg_data'}), 400

        eeg_array = np.array(data['eeg_data'], dtype=np.float32)  # shape: (channels, time)
        eeg_tensor = torch.from_numpy(eeg_array).unsqueeze(0).to(device)  # shape: (1, C, T)

        with torch.no_grad():
            output = model(eeg_tensor)
            pred = torch.argmax(output, dim=1).item()

        return jsonify({'prediction': 'left' if pred == 0 else 'right'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
