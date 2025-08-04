import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from plot_utils import plot_roc_curve, plot_precision_recall_curve, plot_threshold_metrics, plot_loss_curve
from data_utils import load_data
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load data
X, y = load_data()

def train_and_test_model():

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # stratify ensures that the split maintains the same proportion of classes
    )

    # Converting the data to pytorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Creating training and testing data sets which related train and test tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Creating data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    class FireRiskModel(nn.Module):
        # Define the neural network architecture
        def __init__(self):
            # Initialize the model
            super(FireRiskModel, self).__init__()
            self.fc1 = nn.Linear(X_train_tensor.shape[1], 64) # Hidden layer 1: 64 neurons
            self.fc2 = nn.Linear(64, 32) # Hidden layer 2: 32 neurons
            self.fc3 = nn.Linear(32, 1) # Output layer: 1 neuron (binary classification)

        # Define the forward pass
        def forward(self, x):
            x = torch.relu(self.fc1(x)) # Activation function for input layer - hidden layer 1
            x = torch.relu(self.fc2(x)) # Activation function for hidden layer 1 - hidden layer 2
            x = torch.sigmoid(self.fc3(x)) # Activation function for hidden layer 2 - output layer
            return x

    # Initialize the model, loss function, and optimizer
    model = FireRiskModel()
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer

    epochs = 50
    epoch_losses = []

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()

        # Keep track of the running loss for each epoch
        running_loss = 0.0

        # Iterate over the training data
        for inputs, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()   

            # Forward pass
            # Compute the model outputs
            outputs = model(inputs) 

            # Compute the loss
            loss = criterion(outputs, labels) 

            # Backward pass
            # Compute the gradients
            loss.backward() 

            # Update the weights
            optimizer.step() 

            # Accumulate the loss
            running_loss += loss.item() # Accumulate the loss


        # print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

        # Compute the average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        # Append the average loss to the list
        epoch_losses.append(avg_loss)

    # Plot loss vs epochs
    plot_loss_curve(epoch_losses, title="MLP Model Loss vs Epochs")

    # Evaluate the model on the test set
    # Set the model to evaluation mode
    model.eval()
    y_preds = []
    y_true = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate over the test data
        for inputs, labels in test_loader:
            # Forward pass
            # Compute the model outputs
            outputs = model(inputs)

            # Append the predictions to the lists
            y_preds.extend(outputs.squeeze().tolist())

            # Append the true labels to the lists
            y_true.extend(labels.squeeze().tolist())

    # Apply a threshold to convert probabilities to binary predictions
    y_preds_binary = [1 if p > 0.5 else 0 for p in y_preds]

    print("MLP Classifier Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_preds_binary))

    print("\nClassification Report:")
    print(classification_report(y_true, y_preds_binary))

    auc_score = roc_auc_score(y_true, y_preds)
    print(f"ROC AUC Score: {auc_score:.4f}")

    # Plot the ROC curve
    plot_roc_curve(y_true, y_preds, title="MLP Model ROC Curve")

    # # Precision-Recall Curve
    # plot_precision_recall_curve(y_true, y_preds, title="MLP Model Precision-Recall Curve")
    
    # Plot Precision, Recall, and F1 Score against different thresholds
    plot_threshold_metrics(y_true, y_preds, title="MLP Model Threshold Metrics")

    # return the results for models comparison
    results = ("MLP model", y_test, y_preds)
    return results




if __name__ == "__main__":
    train_and_test_model()