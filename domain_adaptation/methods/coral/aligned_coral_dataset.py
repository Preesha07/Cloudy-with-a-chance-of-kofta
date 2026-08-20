import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

Image.MAX_IMAGE_PIXELS = None

class AlignedCoralDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root_sentinel = Path(opt.dataroot)
        self.root_liss4 = Path(opt.dataroot_liss4) if getattr(opt, 'dataroot_liss4', None) else None
        self.fine_size = getattr(opt, 'fineSize', 256)

        if (self.root_sentinel / 'trainA').exists():
            self.dir_A = self.root_sentinel / 'trainA'
            self.dir_B = self.root_sentinel / 'trainB'
        else:
            self.dir_A = self.root_sentinel / 'A'
            self.dir_B = self.root_sentinel / 'B'

        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        self.paths_A = sorted([p for p in self.dir_A.glob('*') if p.suffix.lower() in valid_exts])
        self.paths_B = sorted([p for p in self.dir_B.glob('*') if p.suffix.lower() in valid_exts])

        assert len(self.paths_A) > 0, f"No images found in {self.dir_A}"

        self.paths_liss4 = []
        if self.root_liss4 and self.root_liss4.exists():
            raw_jpgs = list(self.root_liss4.rglob('*.jpg'))
            for p in raw_jpgs:
                if p.is_file() and not p.name.startswith('.'):
                    self.paths_liss4.append(p)
            self.paths_liss4 = sorted(self.paths_liss4)

        self.transform_sentinel = transforms.Compose([
            transforms.Resize((self.fine_size, self.fine_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _get_random_crop(self, img):
        w, h = img.size
        if w < self.fine_size or h < self.fine_size:
            img = img.resize((max(w, self.fine_size), max(h, self.fine_size)), Image.BICUBIC)
            w, h = img.size

        x = random.randint(0, w - self.fine_size)
        y = random.randint(0, h - self.fine_size)
        crop = img.crop((x, y, x + self.fine_size, y + self.fine_size))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(crop)

    def __getitem__(self, index):
        path_A = self.paths_A[index]
        path_B = self.paths_B[index]

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')

        tensor_A = self.transform_sentinel(img_A)
        tensor_B = self.transform_sentinel(img_B)

        item = {
            'A': tensor_A,
            'B': tensor_B,
            'C': tensor_A,
            'A_paths': str(path_A),
            'B_paths': str(path_B),
            'C_paths': str(path_A)
        }

        if self.paths_liss4:
            # Fallback loop: skip corrupt files and load next available image
            for offset in range(15):
                idx = (index + offset) % len(self.paths_liss4)
                path_liss4 = self.paths_liss4[idx]
                try:
                    img_liss4 = Image.open(path_liss4).convert('RGB')
                    item['LISS4'] = self._get_random_crop(img_liss4)
                    item['LISS4_paths'] = str(path_liss4)
                    break
                except Exception:
                    continue

        return item

    def __len__(self):
        return len(self.paths_A)
