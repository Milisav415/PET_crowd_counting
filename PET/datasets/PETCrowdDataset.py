import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms as standard_transforms
from torch.utils.data import Dataset
from PIL import Image



def load_data(img_path, points):
    img = Image.open(img_path).convert("RGB")
    return img, points


def random_crop(img, points, patch_size=(1920, 1080)):
    crop_w, crop_h = patch_size
    _, H, W = img.shape
    if H < crop_h or W < crop_w:
        start_h, start_w = 0, 0
    else:
        start_h = random.randint(0, H - crop_h)
        start_w = random.randint(0, W - crop_w)
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    cropped_img = img[:, start_h:end_h, start_w:end_w]

    points = np.array(points, dtype=float)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    mask = (points_tensor[:, 0] >= start_w) & (points_tensor[:, 0] <= end_w) & \
           (points_tensor[:, 1] >= start_h) & (points_tensor[:, 1] <= end_h)
    cropped_points = points_tensor[mask].clone()
    if cropped_points.nelement() != 0:
        cropped_points[:, 0] -= start_w
        cropped_points[:, 1] -= start_h
    else:
        cropped_points = torch.empty((0, 2), dtype=torch.float32)

    return cropped_img, cropped_points.numpy()


class PETCrowdDataset(Dataset):
    def __init__(self, data_root, annotations_file, transform=None, train=True, flip=False, patch_size=(1920, 1080),
                 split_ratio=0.8):
        self.data_root = data_root
        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size

        # Load annotations from JSON (format: { "img1.jpg": [[x, y], ...], ... })
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        all_images = sorted(list(self.annotations.keys()))
        split_index = int(len(all_images) * split_ratio)
        self.img_list = all_images[:split_index] if self.train else all_images[split_index:]
        self.nSamples = len(self.img_list)

    def compute_density(self, points):
        if len(points) == 0:
            return torch.tensor(999.0).reshape(-1)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < self.nSamples, "index range error"
        img_file = self.img_list[index]
        img_path = os.path.join(self.data_root, img_file)
        points = np.array(self.annotations[img_file], dtype=float)

        img, points = load_data(img_path, points)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.train:
            scale_range = [0.8, 1.2]
            _, H, W = img.shape
            patch_w, patch_h = self.patch_size  # patch_size is now (width, height)
            scale = random.uniform(*scale_range)
            # Ensure both dimensions are larger than the patch dimensions after scaling
            if scale * H > patch_h and scale * W > patch_w:
                img = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(
                    0)
                points *= scale

        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)

        if self.train and self.flip and random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            if len(points) > 0:
                # Adjust the x-coordinate based on the patch width
                points[:, 0] = self.patch_size[0] - points[:, 0]

        target = {}
        target['points'] = torch.tensor(points, dtype=torch.float32)
        target['labels'] = torch.ones(len(points), dtype=torch.long)
        if self.train:
            target['density'] = self.compute_density(points)
        else:
            target['image_path'] = img_path

        return img, target


def build(image_set, args):
    # Define your transforms. You might have separate transforms for training and validation.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Path to your annotations JSON file
    ann_file = os.path.join(args.data_path, 'annotations.json')

    if image_set == 'train':
        dataset = PETCrowdDataset(args.data_path, ann_file, transform=train_transform,
                                  train=True, flip=True)
    elif image_set == 'val':
        dataset = PETCrowdDataset(args.data_path, ann_file, transform=val_transform,
                                  train=False)
    else:
        raise ValueError(f'Unknown image_set {image_set}')

    return dataset