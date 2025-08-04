import torch
import torch.nn as nn
from data_utils import load_data

# Load data
X, _ = load_data()

# Convert to tensor
X_tensor = torch.tensor(X.values, dtype=torch.float32)

class FireRiskModel(nn.Module):
    def __init__(self):
        super(FireRiskModel, self).__init__()
        self.fc1 = nn.Linear(X_tensor.shape[1], 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))  # Outputs in [0, 1]
        return x

# Instantiate and load best trained model
model = FireRiskModel()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Run predictions
with torch.no_grad():
    outputs = model(X_tensor).squeeze().numpy()

# Convert to percentage and label
for i, prob in enumerate(outputs):
    risk_pct = prob * 100

    if risk_pct <= 20:
        risk_level = "Low"
    elif risk_pct <= 40:
        risk_level = "Medium"
    elif risk_pct <= 75:
        risk_level = "High"
    else:
        risk_level = "Very High"

    print(f"Sample {i+1}: Fire Risk Score = {risk_pct:.2f}% → {risk_level}")
