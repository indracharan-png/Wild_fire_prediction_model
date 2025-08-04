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

    # First split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y # stratify ensures that the split maintains the same proportion of classes
    )
    # Second split: 80% of 80% = 72% train, 18% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Converting the data to pytorch tensors
    # train tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    # test tensors
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    # validation tensors
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)


    # Creating training and testing data sets which related train and test tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Creating data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    class FireRiskModel(nn.Module):
        # Define the neural network architecture
        def __init__(self):
            # Initialize the model
            super(FireRiskModel, self).__init__()
            self.fc1 = nn.Linear(X_train_tensor.shape[1], 128) # Input layer (13 features) - Hidden layer 1(128 neurons)
            self.bn1 = nn.BatchNorm1d(128) # Batch normalization for hidden layer 1

            self.fc2 = nn.Linear(128, 64) # Hidden layer 1(128 neurons) - Hidden layer 2(64 neurons)
            self.bn2 = nn.BatchNorm1d(64) # Batch normalization for hidden layer 2

            self.fc3 = nn.Linear(64, 32) # Hidden layer 2(64 neurons) - Hidden layer 3(32 neurons)
            self.bn3 = nn.BatchNorm1d(32) # Batch normalization for hidden layer 3

            self.fc4 = nn.Linear(32, 1) # Hidden layer 3(32 neurons) - Output layer (1 neuron - binary classification)

            # Adding dropout layers to prevent overfitting
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.2)

        # Define the forward pass
        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x))) # Activation function for input layer - hidden layer 1
            x = self.dropout1(x)

            x = torch.relu(self.bn2(self.fc2(x))) # Activation function for hidden layer 1 - hidden layer 2
            x = self.dropout2(x)

            x =  torch.relu(self.bn3(self.fc3(x))) # Activation function for hidden layer 2 - hidden layer 3
            x = self.dropout3(x)

            x = torch.sigmoid(self.fc4(x)) # Activation function for hidden layer 2 - output layer
            return x

    # Initialize the model, loss function, and optimizer
    model = FireRiskModel()
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    epochs = 80
    epoch_losses = []

    # Stopping criteria for early stopping
    best_val_loss = float('inf')
    patience = 25
    counter = 0

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

        # Compute the average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        # Append the average loss to the list
        epoch_losses.append(avg_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)


        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save model if desired
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

    # Plot loss vs epochs
    plot_loss_curve(epoch_losses, title="MLP Model Loss vs Epochs")


    # Evaluate the model on the test set
    # Load the best model
    model.load_state_dict(torch.load("best_model.pt"))
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
    y_preds_binary = [1 if p > 0.4 else 0 for p in y_preds]


    print("Optimized MLP Classifier Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_preds_binary))

    print("\nClassification Report:")
    print(classification_report(y_true, y_preds_binary))

    auc_score = roc_auc_score(y_true, y_preds)
    print(f"ROC AUC Score: {auc_score:.4f}")

    # Plot the ROC curve
    plot_roc_curve(y_true, y_preds, title="Optimized MLP Model ROC Curve")

    # # Precision-Recall Curve
    # plot_precision_recall_curve(y_true, y_preds, title="MLP Model Precision-Recall Curve")
    
    # Plot Precision, Recall, and F1 Score against different thresholds
    plot_threshold_metrics(y_true, y_preds, title="Optimized MLP Model Threshold Metrics")

    # return the results for models comparison
    results = ("MLP model", y_test, y_preds)
    return results




if __name__ == "__main__":
    train_and_test_model()