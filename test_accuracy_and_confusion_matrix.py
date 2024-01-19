import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix


# Bruger gpu hivs tilgængelig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.RandomAffine(5, translate=(0.1, 0.1), scale=(0.1, 0.1), shear=0.1),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(root='kaggle_dataset\\test', transform=transform_test)  
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True) 

# Laver CNN class igen og opbygger arkitekturen
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 73 * 73, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.dropout(x)
        x = self.pool1(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.dropout(x)
        x = self.pool2(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Indlæser model med bedste validation accuracy
best_model_path = 'best_model_kaggle.pth' #på store dataset er det 'final models\dropout0.0\best_model_(57).pth'
best_model = CNN().to(device) #kører på gpu
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval() #evaluation mode
class_labels = test_dataset.classes

# Printer class labels
# print(f"Class 0 (Negative): {class_labels[0]}")
# print(f"Class 1 (Positive): {class_labels[1]}")

# Tester modellens accuracy på testsættet
with torch.no_grad():
    correct_test = 0
    total_test = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = torch.round(best_model(inputs))
        correct_test += torch.sum(torch.eq(predicted, labels.view_as(predicted))).item()

# Beregner test accuracy
accuracy_test = correct_test / len(test_loader.dataset)
print(f'Test Accuracy: {accuracy_test:.4f}')


# =======================================
# Laver nu en confusion matrix
from sklearn.metrics import confusion_matrix
import torch
import numpy as np

best_model.eval() #evaluation mode

# Laver lister til confusion matrix
all_labels = []
all_predictions = []

# Går gennem alle billeder og labels
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = torch.round(best_model(inputs)) # her beregnes predriction for hvert billede

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Laver det i NumPy
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Bruger funktionen confusion matrix til at gøre det
cm = confusion_matrix(all_labels, all_predictions)

# Printer vores confusion matrix
print("Confusion Matrix:")
print(cm)