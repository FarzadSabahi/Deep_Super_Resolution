import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image
import torchvision.transforms as transforms

# Define a simple CNN model
class SuperResolutionModel(nn.Module):
    def __init__(self):
        super(SuperResolutionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # We want the output to retain the image's color channels, so no ReLU here
        return x

# Create a model instance
model = SuperResolutionModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare data
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # resizing to low resolution
    transforms.ToTensor(),
])

# We'll use CIFAR10 for simplicity, you should use a suitable dataset for super-resolution tasks
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training process
num_epochs = 5
for epoch in range(num_epochs):
    for data in loader:
        inputs, _ = data  # we do not need the dataset's labels

        # Generate high-resolution images (targets)
        high_res_targets = ToTensor()(ToPILImage()(inputs.squeeze(0)).resize((128, 128), Image.BICUBIC)).unsqueeze(0)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, high_res_targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print('Training finished.')
