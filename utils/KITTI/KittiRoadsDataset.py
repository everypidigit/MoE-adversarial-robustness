import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os


class KittiRoadsDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(256, 256)):
        """
        Args:
            root_dir (string): Directory with all images and masks.
            transform (callable, optional): Transformations applied to the image.
            image_size (tuple): Target size for images and masks.
        """  # noqa: E501
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.image_dir = os.path.join(root_dir, 'image_2')
        self.gt_dir = os.path.join(root_dir, 'gt_image_2')
        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]  # noqa: E501
        )

        self.color_map = {
            (255, 0, 0): 0,  # Red -> Class 0 (Background)
            (0, 0, 0): 1,     # Black -> Class 1 (Obstacle)
            (255, 0, 255): 2  # Magenta -> Class 2 (Road)
        }

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def mask_to_class_index(self, mask):
        """Convert RGB mask to class index tensor."""
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask_array = np.array(mask)
        class_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)

        for color, class_index in self.color_map.items():
            class_mask[(mask_array == color).all(axis=2)] = class_index

        return torch.tensor(class_mask, dtype=torch.long)  # Convert to tensor

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        road_type, image_index = image_name.split('_')[0], image_name.split('_')[1]  # noqa: E501
        road_gt_filename = f"{road_type}_road_{image_index}"
        road_gt_path = os.path.join(self.gt_dir, road_gt_filename)

        if os.path.exists(road_gt_path):
            road_gt = Image.open(road_gt_path).convert('RGB')
            road_gt = self.mask_to_class_index(road_gt)
        else:
            print(f"Warning: Road GT missing for {image_name}, using blank mask.")  # noqa: E501
            road_gt = torch.zeros(self.image_size, dtype=torch.long)

        image = self.image_transform(image)

        return {'image': image, 'road_gt': road_gt}
