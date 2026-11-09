import os
from torch.utils.data import Dataset
from PIL import Image
import torch 
import numpy as np
import torch.nn.functional as F

class FlSeaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.tif'):
                    if 'depth' in root:
                        depth_path = os.path.join(root, file)
                        image_path = os.path.join(root.replace('depth', 'imgs'), file.replace('_SeaErra_abs_depth.tif', '.tiff'))
                        data.append((image_path, depth_path))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, depth_path = self.data[idx]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        return image, depth

class USODDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.images = []
            self.depths = []

            # Loop through each subfolder
            for subfolder in os.listdir(root_dir):
                subfolder_path = os.path.join(root_dir, subfolder)
                if os.path.isdir(subfolder_path):
                    # Check if the subfolder contains 'depth' and 'RGB' folders
                    if all([os.path.isdir(os.path.join(subfolder_path, folder)) for folder in ['depth', 'RGB']]):
                        # Load images and depths from the subfolders
                        images_folder = os.path.join(subfolder_path, 'RGB')
                        depth_folder = os.path.join(subfolder_path, 'depth')
                        images_list = sorted(os.listdir(images_folder))
                        depth_list = sorted(os.listdir(depth_folder))

                        for img_name, depth_name in zip(images_list, depth_list):
                            img_path = os.path.join(images_folder, img_name)
                            depth_path = os.path.join(depth_folder, depth_name)
                            self.images.append(img_path)
                            self.depths.append(depth_path)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            depth_path = self.depths[idx]

            image = Image.open(img_path)
            depth = Image.open(depth_path)

            if self.transform:
                image = self.transform(image)
                depth = self.transform(depth)

            return image, depth
        
class VAROSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                    if 'A' in root:
                        image_path = os.path.join(root, file)
                        depth_path = os.path.join(root.replace('A', 'D'), file.replace('A', 'D'))
                        data.append((image_path, depth_path))
        return data

    def __len__(self):
        return len(self.data)
    
    def min_max_normalize(self,image):
    # Get the minimum and maximum pixel values
        min_val = torch.min(image)
        max_val = torch.max(image)

        # Normalize the image
        normalized_image = (image - min_val) / (max_val - min_val)
        return normalized_image

    def __getitem__(self, idx):
        image_path, depth_path = self.data[idx]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
            depth = self.min_max_normalize(depth)
            depth = 1 - depth   
        return image, depth




class UVMultiDataset(Dataset):
    def __init__(self, root_dir, transform=None, depth_folder="depth"):
        self.transform = transform
        self.data = []
        root_dir = "C:/Users/yoame/Documents/ML/underwater/UVmulti/all_vid"
        for video in sorted(os.listdir(root_dir)):
            img_dir = os.path.join(root_dir, video, "images")
            dep_dir = os.path.join(root_dir, video, depth_folder)
            if not (os.path.isdir(img_dir) and os.path.isdir(dep_dir)):
                continue
            depths = {os.path.splitext(f)[0]: os.path.join(dep_dir, f)
                      for f in os.listdir(dep_dir)}
            for f in sorted(os.listdir(img_dir)):
                stem = os.path.splitext(f)[0]
                if stem in depths:  # only keep frames that have a depth map
                    self.data.append((os.path.join(img_dir, f), depths[stem]))

    def __len__(self):
        return len(self.data)

    def _load_depth(self, path):
        if path.endswith(".npy"):
            d = np.load(path).astype(np.float32)
        else:
            d = np.array(Image.open(path)).astype(np.float32)
        if d.ndim == 3:          # in case it's saved as 3-channel
            d = d[..., 0]
        return torch.from_numpy(d).unsqueeze(0)   # (1, H, W)

    def __getitem__(self, idx):
        image_path, depth_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        depth = self._load_depth(depth_path)

        if self.transform:
            image = self.transform(image)                 # -> (3, 518, 518)
        # resize depth to match the image
        depth = F.interpolate(depth.unsqueeze(0), size=image.shape[-2:],
                              mode="bilinear", align_corners=False).squeeze(0)

        # same normalisation as VAROS
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = 1 - depth
        return image, depth