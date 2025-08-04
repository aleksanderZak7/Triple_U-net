import os
import cv2
import torch
import numpy as np
import skimage.io as io

import model
import Config
from utils import rtime_print
from data import DataGenerator


class TestModel(object):
    def __init__(self, conf: Config.config) -> None:
        super(TestModel, self).__init__()
        self.conf = conf
        self.datagen = DataGenerator(conf.test_data_path)
        
        self.net = model.net()
        self.net = torch.load(conf.model_path, weights_only=False)
        self.net = self.net.cuda()


    def test(self) -> None:
        for index in range(len(self.datagen)):
            rtime_print('{}/{}'.format(index + 1, len(self.datagen)))
            img, file, HE = self.datagen[index]
            
            # HE channel hematoxylin-eosin staining
            img = torch.unsqueeze(img.cuda(), 0)
            HE = torch.unsqueeze(HE.cuda(), 0)  # type: ignore

            pred = self.predition(img, HE)
            io.imsave(os.path.join(self.conf.save_path, file), pred)


    def predition(self, img, HE) -> np.ndarray:
        with torch.no_grad():
            nuclei, H, RGB = self.net(img, HE)
            
            H = torch.squeeze(H.cpu())
            RGB = torch.squeeze(RGB.cpu())
            nuclei = torch.squeeze(nuclei.cpu())
            
            H = np.array(H * 255, dtype='uint8')
            RGB = np.array(RGB * 255, dtype='uint8')
            nuclei = np.array((nuclei > self.conf.cutoff) * 255, dtype='uint8')

        return cv2.bitwise_or(cv2.bitwise_or(H, RGB), nuclei)