# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model
![image](https://github.com/user-attachments/assets/ad7d40b7-d4c8-4487-a4a7-3bf5bca6d13b)
## DESIGN STEPS

### STEP 1: Problem Statement
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a Convolutional Neural Network (CNN).

### STEP 2: Dataset Collection
Use the Fashion-MNIST dataset, which contains 60,000 training images and 10,000 test images of various clothing items.

### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP 4: Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers to extract features and classify clothing items.

### STEP 5: Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 6: Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM

### Name:ajina joshpin
### Register Number: 212223230008
```python
class CNNClassifier (nn.Module):
  def __init__(self):
    super (CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn. Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn. MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn. Linear (128 * 3 * 3, 128)
    self.fc2 = nn. Linear (128, 64)
    self.fc3= nn. Linear (64, 10)
  def forward(self, x):
    x = self.pool (torch.relu(self.conv1(x)))
    x = self.pool (torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size (0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name:        ')
    print('Register Number:       ')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:        ')
    print('Register Number:       ')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name:ajina joshpin      ')
    print('Register Number: 212223230008')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

```

## OUTPUT
### Training Loss per Epoch
### Confusion Matrix
![image](https://github.com/user-attachments/assets/8f66d75d-15ee-4e1d-bbae-563949c878de)

### Classification Report
![image](https://github.com/user-attachments/assets/0b31a9ce-fec7-4c62-8a5a-ea9628bb91fa)
### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/d20b7b5f-8b19-49d6-ab78-3a3ba2a6db1b)


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
