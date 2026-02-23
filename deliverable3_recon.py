import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReconstructionCNN(nn.Module):
    def __init__(self, action_size=4):
        super(ReconstructionCNN, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1) # 16x32x32
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 32x16x16
        
        # Bottleneck (Inject Action)
        self.flatten_dim = 32 * 16 * 16
        self.fc_encode = nn.Linear(self.flatten_dim + action_size, self.flatten_dim)
        
        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # 16x32x32
        self.dec_conv2 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)  # 3x64x64

    def forward(self, img, action):
        # Encode
        x = F.relu(self.enc_conv1(img))
        x = F.relu(self.enc_conv2(x))
        
        # Flatten and concat action
        x_flat = x.view(x.size(0), -1)
        x_fused = torch.cat((x_flat, action), dim=1)
        
        # Process bottleneck and reshape back to spatial dimensions
        x_bottleneck = F.relu(self.fc_encode(x_fused))
        x_reshaped = x_bottleneck.view(x.size(0), 32, 16, 16)
        
        # Decode
        x_out = F.relu(self.dec_conv1(x_reshaped))
        x_out = torch.sigmoid(self.dec_conv2(x_out)) # Assuming normalized image [0, 1]
        return x_out

def train(model, train_loader, criterion, optimizer, epochs=10, device='cpu'):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        # targets here are img_after
        for imgs_before, actions, imgs_after in train_loader:
            imgs_before, actions = imgs_before.to(device), actions.to(device)
            imgs_after = imgs_after.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs_before, actions)
            loss = criterion(outputs, imgs_after)
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
        for imgs_before, actions, imgs_after in test_loader:
            imgs_before, actions = imgs_before.to(device), actions.to(device)
            imgs_after = imgs_after.to(device)
            
            outputs = model(imgs_before, actions)
            loss = criterion(outputs, imgs_after)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    return avg_test_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReconstructionCNN().to(device)
    criterion = nn.MSELoss() # Pixel-wise MSE
    optimizer = optim.Adam(model.parameters(), lr=0.001)
              
