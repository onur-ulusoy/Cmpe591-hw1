import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PositionMLP(nn.Module):
    def __init__(self, img_channels=3, img_size=64, action_size=4, hidden_size=256):
        super(PositionMLP, self).__init__()
        self.input_dim = img_channels * img_size * img_size
        self.fc1 = nn.Linear(self.input_dim + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2) # Predicts x, y

    def forward(self, img, action):
        # Flatten image: (batch_size, channels * height * width)
        img_flat = img.view(img.size(0), -1) 
        # Concatenate flattened image and one-hot action vector
        x = torch.cat((img_flat, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
    # Dummy initialization to verify it runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PositionMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train(model, train_loader, criterion, optimizer, epochs=20, device=device)
    # test(model, test_loader, criterion, device=device)
