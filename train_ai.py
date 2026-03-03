import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# --- 1. The Neural Architecture ---
class SimpleBrain(nn.Module):
    def __init__(self):
        super(SimpleBrain, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_layers(x)

# --- 2. Training Logic ---
print("--- 1. Loading MNIST Data for Python 3.14 ---")
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

model = SimpleBrain()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print("--- 2. Training the PyTorch Brain ---")
for epoch in range(3):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Complete.")

torch.save(model.state_dict(), "pytorch_digit_brain.pth")
print("--- SUCCESS: Brain saved as .pth file! ---")