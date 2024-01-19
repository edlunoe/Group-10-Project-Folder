import torch
import torch.nn as nn
from tqdm import tqdm  # Importerer tqdm for progressionsbar under epoker
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Bruger gpu hivs tilgængelig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
# Data Preprocessing defineres for trænings- og valideringsdata
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.RandomAffine(5, translate=(0.1, 0.1), scale=(0.1, 0.1), shear=0.1),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.RandomAffine(5, translate=(0.1, 0.1), scale=(0.1, 0.1), shear=0.1),
    transforms.ToTensor()
])

if __name__ == '__main__': # datapreprocessing
    train_dataset = datasets.ImageFolder(root='combined_data\\train_test_valid\\training', transform=transform_train)  
    val_dataset = datasets.ImageFolder(root='combined_data/train_test_valid/validation', transform=transform_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True) 
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Laver CNN class og opbygger arkitekturen
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
        self.fc1 = nn.Linear(32 * 73 * 73, 128)  # regn efter igen
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout2d(0.0)

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

# Initialiserer modellen med Adam-optimizer og loss-funktionen binary cross entropy
cnn = CNN().to(device)
optimizer = optim.Adam(cnn.parameters())
criterion = nn.BCELoss()

# Træner CNN
def train():
    num_epochs = 50
    for epoch in range(num_epochs):
        cnn.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # bruger tqdm til progressionsbar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = criterion(outputs, labels.float().view(-1, 1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Beregner training accuracy
                predicted = torch.round(outputs)
                total_train += labels.size(0)
                correct_train += (predicted == labels.float().view(-1, 1)).sum().item()

                pbar.update(1)
                pbar.set_postfix({'Loss': running_loss / (pbar.n + 1), 'Accuracy': correct_train / total_train})

        # beregner gennemsnitlig training accuracy for epoken
        average_train_accuracy = correct_train / total_train

        # Evaluerer på valideringssæt
        cnn.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = cnn(inputs)
                predicted = torch.round(outputs)
                total_val += labels.size(0)
                correct_val += (predicted == labels.float().view(-1, 1)).sum().item()

        # Beregner validation accuracy
        accuracy_val = correct_val / total_val

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
            f'Training Accuracy: {average_train_accuracy:.4f}, Validation Accuracy: {accuracy_val:.4f}')
        torch.save(cnn.state_dict(), f'model_{epoch+1}.pth')

if __name__ == '__main__':
    train()        
