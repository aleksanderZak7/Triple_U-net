import os
import torch
import skimage
import numpy as np
from typing import Tuple
from utils import separate_stain
from torch.utils.data.dataset import Dataset


class DataGenerator(Dataset):
    def __init__(self, data_dir) -> None:
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.img_ids = sorted(os.listdir(self.data_dir))  
        
        
    def load_image(self, index) -> Tuple[torch.FloatTensor, str]:
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.data_dir, img_id)
        
        img = skimage.io.imread(imgFile)
        return torch.FloatTensor(np.transpose(img, (2, 0, 1)) / 255), img_id
        
        
    def load_HE(self, index) -> torch.FloatTensor:
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.data_dir, img_id)
        
        img = skimage.io.imread(imgFile)
        
        HE = separate_stain(img)
        output = torch.FloatTensor(np.transpose(HE, (2, 0, 1)) / 255)
        
        return output
        
        
    def __getitem__(self, item) -> Tuple[torch.FloatTensor, str, torch.FloatTensor]:
        img, id = self.load_image(item)
        HE = self.load_HE(item)
        
        return img, id, HE
        
        
    def __len__(self) -> int:
        return len(self.img_ids)