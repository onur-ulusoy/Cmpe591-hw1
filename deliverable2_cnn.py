import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import get_loaders

class PositionCNN(nn.Module):
    def __init__(self):
        super(PositionCNN, self).__init__()
        # Input: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # -> 16 x 64 x 64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # -> 32 x 32 x 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> 64 x 16 x 16
        
        self.flatten_dim = 64 * 16 * 16
        self.fc1 = nn.Linear(self.flatten_dim + 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, img, action):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = torch.cat((x, action), dim=1)
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for img_b, action, target_pos, _ in train_loader:
            img_b, action, target_pos = img_b.to(device), action.to(device), target_pos.to(device)
            
            optimizer.zero_grad()
            outputs = model(img_b, action)
            loss = criterion(outputs, target_pos)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
    return loss_history

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for img_b, action, target_pos, _ in test_loader:
            img_b, action, target_pos = img_b.to(device), action.to(device), target_pos.to(device)
            outputs = model(img_b, action)
            loss = criterion(outputs, target_pos)
            test_loss += loss.item()
            
    avg_loss = test_loss / len(test_loader)
    print(f"Test MSE Loss: {avg_loss:.6f}")
    return avg_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(batch_size=32)

    model = PositionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = train(model, train_loader, criterion, optimizer, epochs=20, device=device)
    test(model, test_loader, criterion, device=device)

    plt.plot(losses)
    plt.title("CNN Position Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("cnn_loss_curve.png")
    plt.show()
