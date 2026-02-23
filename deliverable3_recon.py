import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_loaders

class ReconstructionNet(nn.Module):
    def __init__(self):
        super(ReconstructionNet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) # -> 32 x 64 x 64
        self.enc2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) # -> 64 x 32 x 32
        self.enc3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # -> 128 x 16 x 16
        
        # Bottleneck processing (Action injection)
        # We will concatenate action as channels
        self.btlnk = nn.Conv2d(128 + 4, 128, 3, padding=1)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # -> 64 x 32 x 32
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) # -> 32 x 64 x 64
        self.dec3 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)  # -> 3 x 128 x 128

    def forward(self, img, action):
        # Encoder
        e1 = F.relu(self.enc1(img))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2)) # Shape: (B, 128, 16, 16)
        
        # Expand action to match spatial dimensions (16x16)
        # Action (B, 4) -> (B, 4, 16, 16)
        action_map = action.view(action.size(0), 4, 1, 1).expand(-1, -1, 16, 16)
        
        # Concatenate
        x = torch.cat((e3, action_map), dim=1)
        x = F.relu(self.btlnk(x))
        
        # Decoder
        d1 = F.relu(self.dec1(x))
        d2 = F.relu(self.dec2(d1))
        out = torch.sigmoid(self.dec3(d2)) # Output in [0, 1]
        return out

def train(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for img_b, action, _, img_a in train_loader:
            img_b, action, img_a = img_b.to(device), action.to(device), img_a.to(device)
            
            optimizer.zero_grad()
            outputs = model(img_b, action)
            loss = criterion(outputs, img_a)
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
        for img_b, action, _, img_a in test_loader:
            img_b, action, img_a = img_b.to(device), action.to(device), img_a.to(device)
            outputs = model(img_b, action)
            loss = criterion(outputs, img_a)
            test_loss += loss.item()
            
    avg_loss = test_loss / len(test_loader)
    print(f"Test MSE Loss: {avg_loss:.6f}")
    return avg_loss

def save_comparison(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        img_b, action, _, img_a = next(iter(test_loader))
        img_b, action, img_a = img_b.to(device), action.to(device), img_a.to(device)
        recon = model(img_b, action)
        
        # Convert to CPU numpy for plotting
        img_gt = img_a[0].permute(1, 2, 0).cpu().numpy()
        img_pred = recon[0].permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth")
        plt.imshow(img_gt)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Reconstruction")
        plt.imshow(img_pred)
        plt.axis('off')
        
        plt.savefig("reconstruction_comparison.png")
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(batch_size=32)

    model = ReconstructionNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = train(model, train_loader, criterion, optimizer, epochs=30, device=device)
    test(model, test_loader, criterion, device=device)
    save_comparison(model, test_loader, device)

    plt.figure()
    plt.plot(losses)
    plt.title("Reconstruction Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("recon_loss_curve.png")
    plt.show()
