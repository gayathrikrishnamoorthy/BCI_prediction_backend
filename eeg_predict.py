import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Load test data
X = np.load('X.npy', allow_pickle=True)  # shape: (samples, channels, time)
y = np.load('y.npy', allow_pickle=True)

# Define the CNN model (same as training)
class EEGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(X.shape[1], 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d((1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (batch, 1, channels, time)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)  # flatten all except batch

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 64).to(x.device)
            self.fc2 = nn.Linear(64, 2).to(x.device)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = EEGClassifier().to(device)

# Run dummy forward pass to create fc1 and fc2
dummy_input = torch.randn(1, X.shape[1], X.shape[2]).to(device)
model(dummy_input)

# Load the saved model weights
model.load_state_dict(torch.load("eeg_model.pth", map_location=device))
model.eval()

# Predict function
def predict(model, X_sample):
    model.eval()
    X_tensor = torch.from_numpy(X_sample.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(X_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "left" if pred == 0 else "right"

# Run prediction for first 5 samples
for i in range(5):
    sample = X[i]
    prediction = predict(model, sample)
    print(f"Sample {i+1} prediction: {prediction}")
