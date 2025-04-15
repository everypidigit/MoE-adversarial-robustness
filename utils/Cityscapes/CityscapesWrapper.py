from torch.utils.data import Dataset
from torchvision import transforms


class CityscapesWrapper(Dataset):
    def __init__(self, base_dataset, target_transform=None):
        self.base = base_dataset
        self.target_transform = target_transform
        self.to_tensor = transforms.PILToTensor()

    def __getitem__(self, idx):
        img, target = self.base[idx]
        if self.target_transform:
            target = self.target_transform(target)
        target = self.to_tensor(target)
        return img, target

    def __len__(self):
        return len(self.base)
