import os
import cv2
import torch
import skimage
import numpy as np 
from utils import *
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class DataGenerator(Dataset):
    def __init__(self, data_dir, label_dir,edge_dir, transform=None) -> None:
        super(Dataset, self).__init__()
        self.data_dir   =   data_dir
        self.label_dir  =   label_dir
        self.edge_dir  =   edge_dir
        self.transform  =   transform
        self.img_ids    =   sorted(os.listdir(self.data_dir))
        self.error_name =   True
        self.right_name =   False
        
    def just_img_name(self, img_id) -> bool:
        if img_id.find('pre') >= 0 or img_id.find('sep') >= 0 or img_id.find('mask') >= 0:
            return self.error_name
        return self.right_name    
        
    def load_image(self, index, norm=True, no_transpose=False) -> tuple[torch.FloatTensor, str]:
        img_id  = self.img_ids[index]
        if self.just_img_name(img_id):
            return torch.FloatTensor([]), img_id
        imgFile = os.path.join(self.data_dir, img_id)
        img     = skimage.io.imread(imgFile)
        if no_transpose:
            return img, img_id
        if norm:
            return torch.FloatTensor(np.transpose(img, (2,0,1)) / 255), img_id
        return torch.FloatTensor(np.transpose(img, (2,0,1))), img_id
        
    def load_mask(self, index,no_tensor=False) -> torch.FloatTensor:
        mask_ids    = self.img_ids[index]
        imgFile     = os.path.join(self.label_dir, mask_ids)
        mask        = skimage.io.imread(imgFile)

        if no_tensor:
            return mask
        return torch.FloatTensor(mask > 0)

        
    def load_edge(self, index, no_tensor=False) -> torch.FloatTensor:
        feat_ids    = self.img_ids[index]
        
        imgFile     = os.path.join(self.edge_dir, feat_ids)
        mask        = skimage.io.imread(imgFile)

        if no_tensor:
            return mask
        return torch.FloatTensor(mask>0)
        
    def load_HE(self, index, no_transpose=False) -> torch.FloatTensor | np.ndarray:
        img_id  = self.img_ids[index]
        if self.just_img_name(img_id):
            return torch.FloatTensor([])
        
        imgFile = os.path.join(self.data_dir, img_id)
        img     = skimage.io.imread(imgFile)
        
        HE = separate_stain(img)
        #HE      = skimage.color.gray2rgb(HE) # -> HE shape: (64, 64, 3, 3) !!!!!!!!!!!
        
        if no_transpose:
            return HE

        output = torch.FloatTensor(np.transpose(HE, (2, 0, 1))/255)
        return output
        
    def __getitem__(self, item) -> tuple[torch.FloatTensor, torch.FloatTensor | np.ndarray, torch.FloatTensor, torch.FloatTensor]:
        img, _     = self.load_image(item, no_transpose=True)
        masks    = self.load_mask(item, no_tensor=True)
        edge    = self.load_edge(item, no_tensor=True)
        HE      = self.load_HE(item, no_transpose=True)

        if self.transform is not None:
            return self.transform(img, masks, HE, edge)
        
        return img, HE, masks, edge
        
    def __len__(self) -> int:
        return len(self.img_ids)
        
def save_output(save_path,img,name) -> None:
    batch=img.size()[0]
    for i in range(batch):
        imgFile = os.path.join(save_path, name[i])
        cv2.imwrite(imgFile,np.array(img[i,:,:].cpu().detach().numpy())*255)
        
def collater(data) -> tuple[torch.Tensor, ...]:
    HE = []
    img = []
    edge = []
    masks = []

    for sample in data:
        img.append(sample[0])
        HE.append(sample[1])
        edge.append(sample[3])
        masks.append(sample[2])

    HE = torch.stack(HE, 0)
    img = torch.stack(img, 0)
    masks = torch.stack(masks, 0)
    edge = torch.stack(edge, 0)

    return img, HE, masks, edge

def seg_loss(prediction, ground_truth) -> torch.Tensor:
    ground_truth = torch.squeeze(ground_truth.cpu()).view(-1) 
    prediction = torch.squeeze(prediction.cpu()).view(-1)

    loss = nn.BCELoss()(prediction, ground_truth)
    return loss

def soft_dice(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    smooth = 1e-5
    
    input_flat = torch.squeeze(input.cpu())
    target_flat = torch.squeeze(target.cpu())
    
    dim=(1, 2) if len(input_flat.shape) == 3 else (0, 1)
    
    intersection = input_flat * target_flat
    loss = 2. * (torch.sum(intersection,dim) + smooth) / (torch.sum(input_flat*input_flat,dim)\
                    + torch.sum(target_flat*target_flat,dim) + smooth)

    loss = torch.mean(1 - loss)
    return loss
    

def soft_truncate_ce_loss(pre,label, delta=0.2) -> torch.Tensor:
    smooth = 1e-5
    pre = torch.squeeze(pre.cpu())
    label = torch.squeeze(label.cpu())
    condition = label>.5
    ret = torch.where(condition,pre,1-pre)
    ret = torch.log(ret+smooth)
    smooth_truncate = label *(-1/(2*(delta**2))*pre**2 - np.log(delta) + 1/2) \
                      + (1-label) * (-1/(2*(delta**2))*(1-pre)**2 - np.log(delta) + 1/2)
    condition = torch.le(ret,np.log(delta))
    return torch.mean(torch.where(condition, smooth_truncate,-ret))
    
def IOU_loss(pre,label, batch_size) -> torch.Tensor:
    dim=(1, 2)
    pre = torch.squeeze(pre.cpu())
    label = torch.squeeze(label.cpu())
    intersection = pre * label
    union = pre + label -intersection
    IoU = torch.sum(intersection,dim)/torch.sum(union,dim)
    return (1 - torch.mean(IoU)) / batch_size

def FP_loss(pre, label) -> torch.Tensor:
    smooth = 1e-5
    pre = torch.squeeze(pre.cpu())
    label = torch.squeeze(label.cpu())
    
    label = 1 - label
    return torch.mean(label * torch.log(1 - pre + smooth))
    
class LossVariance(nn.Module):
    def __init__(self) -> None:
        super(LossVariance, self).__init__()
        
    def forward(self, input, target) -> torch.Tensor:
        B = input.size(0)

        loss = 0
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]

            sum_var = 0
            for val in unique_vals:
                instance = input[k][:, target[k] == val]
                if instance.size(1) > 1:
                    sum_var = sum_var + torch.sum(torch.var(instance))
            loss = loss + sum_var / (len(unique_vals) + 1e-8)
        loss = loss / B
        return loss.cpu()