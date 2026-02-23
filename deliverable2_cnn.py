import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PositionCNN(nn.Module):
    def __init__(self, action_size=4):
        super(PositionCNN, self).__init__()
        # Assuming input image is 3x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1) # Output: 16x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # Output: 32x16x16
        
        self.flatten_dim = 32 * 16 * 16
        self.fc1 = nn.Linear(self.flatten_dim + action_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, img, action):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(model, train_loader, criterion, optimizer, epochs=10, device='cpu'):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, actions, targets in train_loader:
            imgs, actions, targets = imgs.to(device), actions.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, actions)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")
    return loss_history

def test(model, test_loader, criterion, device='cpu'):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for imgs, actions, targets in test_loader:
            imgs, actions, targets = imgs.to(device), actions.to(device), targets.to(device)
            outputs = model(imgs, actions)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    return avg_test_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PositionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
  
