import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Design the "Brain" (Neural Network)
class DigitBrain(nn.Module):
    def __init__(self):
        super(DigitBrain, self).__init__()
        self.main_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), # 784 pixels to 128 neurons
            nn.ReLU(),            # The "thinking" activation
            nn.Linear(128, 10)    # 10 output neurons (for digits 0-9)
        )

    def forward(self, x):
        return self.main_layers(x)

# 2. Setup Training
print("--- 1. Loading Data for Python 3.14 ---")
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

model = DigitBrain()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. The "Study" Phase
print("--- 2. Training the AI (3 Epochs) ---")
for epoch in range(3):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Complete.")

# 4. Save the Brain
torch.save(model.state_dict(), "brain_314.pth")
print("--- SUCCESS: 'brain_314.pth' saved! ---")