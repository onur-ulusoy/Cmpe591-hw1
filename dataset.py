from torch.utils.data import Dataset, DataLoader
import torch
import glob
import os

class RobotDataset(Dataset):
    def __init__(self, mode='train'):
        # Define the path to the dataset folder
        data_dir = "dataset"
        
        # Load all .pt files from the 'dataset/' directory
        # We use os.path.join to ensure it works on Linux/Mac/Windows
        self.imgs_before = torch.cat([torch.load(f) for f in glob.glob(os.path.join(data_dir, "imgs_before_*.pt"))])
        self.imgs_after = torch.cat([torch.load(f) for f in glob.glob(os.path.join(data_dir, "imgs_after_*.pt"))])
        self.actions = torch.cat([torch.load(f) for f in glob.glob(os.path.join(data_dir, "actions_*.pt"))])
        self.positions = torch.cat([torch.load(f) for f in glob.glob(os.path.join(data_dir, "positions_*.pt"))])

        # Normalize Images to [0, 1]
        self.imgs_before = self.imgs_before.float() / 255.0
        self.imgs_after = self.imgs_after.float() / 255.0

        # Simple split: First 80% train, last 20% test
        split_idx = int(0.8 * len(self.actions))
        
        if mode == 'train':
            self.imgs_before = self.imgs_before[:split_idx]
            self.imgs_after = self.imgs_after[:split_idx]
            self.actions = self.actions[:split_idx]
            self.positions = self.positions[:split_idx]
        else:
            self.imgs_before = self.imgs_before[split_idx:]
            self.imgs_after = self.imgs_after[split_idx:]
            self.actions = self.actions[split_idx:]
            self.positions = self.positions[split_idx:]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # One-hot encode action (0 -> [1,0,0,0])
        action_one_hot = torch.zeros(4)
        action_one_hot[self.actions[idx].long()] = 1.0
        
        return self.imgs_before[idx], action_one_hot, self.positions[idx], self.imgs_after[idx]

def get_loaders(batch_size=32):
    train_set = RobotDataset(mode='train')
    test_set = RobotDataset(mode='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
