import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
import os
import time

# --- 1. THE BRAIN ---
class DigitBrain(nn.Module):
    def __init__(self):
        super(DigitBrain, self).__init__()
        self.main_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.main_layers(x)

# --- 2. THE TRAINING ---
def train_model():
    print("--- Phase 1: Training the Brain ---")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    model = DigitBrain()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"   Step {epoch+1}/3 Finished.")
    
    torch.save(model.state_dict(), "brain_314.pth")
    return model

# --- 3. THE VISION & CAPTURE ---
def run_vision(model):
    if not os.path.exists("captures"):
        os.makedirs("captures")

    cap = cv2.VideoCapture(0)
    print("\n--- Phase 2: Live AI Vision ---")
    print("--- Commands: 'q' to Quit | 's' to Save Screenshot ---")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Focus Box
            cv2.rectangle(frame, (200, 200), (450, 450), (0, 255, 0), 2)
            
            # Preprocessing
            roi = frame[200:450, 200:450]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            inverted = cv2.bitwise_not(resized)
            tensor_img = torch.from_numpy(inverted).float() / 255.0
            input_batch = tensor_img.unsqueeze(0).unsqueeze(0)

            # AI Guess
            with torch.no_grad():
                output = model(input_batch)
                prediction = torch.argmax(output, dim=1).item()
            
            # Text UI
            cv2.putText(frame, f"AI Sees: {prediction}", (210, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('Python 3.14 AI Master', frame)

            # Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"captures/digit_{prediction}_{int(time.time())}.png"
                cv2.imwrite(filename, roi)
                print(f"Saved: {filename}")

    finally:
        print("Safely releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    my_model = DigitBrain()
    if os.path.exists("brain_314.pth"):
        print("Loading existing brain...")
        my_model.load_state_dict(torch.load("brain_314.pth", weights_only=True))
    else:
        my_model = train_model()
    
    my_model.eval()
    run_vision(my_model