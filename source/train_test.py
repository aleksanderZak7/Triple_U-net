import wandb    # !

import os
import time
import data
import torch
import numpy as np
import skimage.io as io

import model
import Config
import transform
from utils import *


class test_model(object):
    def __init__(self, conf: Config.config):
        super(test_model, self).__init__()
        self.conf = conf
        self.datagen = data.trainGenerator(conf.test_data_path, conf.label_path, conf.edg_path)
        
        self.net = model.net()
        self.net.load_state_dict(torch.load(conf.model_path))
        self.net = self.net.cuda()

    def test(self):
        rgb_pre = []
        file_ = []
        HE_pre = []
        nuclei_pre = []
        ground_truth = []
        for index in range(len(self.datagen)):
            rtime_print('{}/{}'.format(index+1, len(self.datagen)))
            img, file = self.datagen.load_image(index)
            if self.datagen.just_img_name(file):
                continue
            
            # HE channel hematoxylin-eosin staining
            HE = self.datagen.load_HE(index)
            img = torch.unsqueeze(img.cuda(), 0)
            HE = torch.unsqueeze(HE.cuda(), 0)  # type: ignore

            mask = self.datagen.load_mask(index)
            nuclei, outh, outrgb = self.predition(img, HE, file)
            rgb_pre.append(torch.squeeze(outrgb.cpu()))
            nuclei_pre.append(nuclei)
            HE_pre.append(torch.squeeze(outh.cpu()))
            ground_truth.append(torch.squeeze(mask))
            file_.append(file)

        evaluation(nuclei_pre, ground_truth, self.conf.cutoff)

    def predition(self, img, HE, file):
        with torch.no_grad():
            nuclei, H, RGB = self.net(img, HE)
            nuclei = torch.squeeze(nuclei.cpu())
            RGB = torch.squeeze(RGB.cpu())
            H = torch.squeeze(H.cpu())
            io.imsave(os.path.join(self.conf.save_path, file[0:(len(
                file)-4)]+'-pre.png'), np.array((nuclei > self.conf.cutoff)*255, dtype='uint8'))
            io.imsave(os.path.join(self.conf.save_path, file[0:(
                len(file)-4)]+'-RGB.png'), np.array(RGB*255, dtype='uint8'))
            io.imsave(os.path.join(self.conf.save_path, file[0:(
                len(file)-4)]+'-H.png'), np.array(H*255, dtype='uint8'))

        return nuclei, H, RGB


class train_model(object):
    def __init__(self, conf):
        super(train_model, self).__init__()
        self.conf = conf
        self.net = model.net().cuda()

        self.optimizer = create_optimizer(conf, self.net)
        self.scheduler = create_lr_scheduler(conf, self.optimizer)

        data_transform = transform.Compose([transform.RandomMirror_h(),
                                            transform.RandomMirror_w(),
                                            transform.rotation(),
                                            transform.flip(),
                                            transform.elastic_transform()])

        self.train_data_Generator = data.trainGenerator(
            self.conf.train_data_path, self.conf.label_path, self.conf.edg_path, data_transform)
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data_Generator,
            batch_size=self.conf.batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=data.collater)

        ####################################
        self.valid_Generator = data.trainGenerator(
            self.conf.valid_data_path, self.conf.label_path, self.conf.edg_path, data_transform)
        
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_Generator,
            batch_size=self.conf.batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=data.collater)

    def epoch_train(self) -> float:
        i: int = 0
        Loss: float = 0.0
        self.net.train()

        for img, H, ground_truth, edge in self.train_loader:
            H = H.cuda()
            img = img.cuda()
            i = i + img.size()[0]
            rtime_print('{}/{}'.format(i, len(self.train_data_Generator)))
            self.optimizer.zero_grad()
            nuclei, contourH, contourRGB = self.net(img, H)
            loss_nuclei = data.soft_dice(nuclei, ground_truth)

            loss_nucleice = data.soft_truncate_ce_loss(nuclei, ground_truth)
            loss_H = data.soft_dice(contourH, edge)
            loss_RGB = data.seg_loss(contourRGB, ground_truth)

            loss = 0.3 * loss_nuclei + loss_nucleice + 0.3 * loss_H + 0.3 * loss_RGB
            loss.backward()
            self.optimizer.step()
            Loss = Loss + loss.item()
        return Loss / len(self.train_data_Generator)

    def epoch_valid(self) -> float:
        i: int = 0
        Loss: float = 0.0
        self.net.eval()

        with torch.no_grad():
            for img, H, ground_truth, edge in self.valid_loader:
                H = H.cuda()
                img = img.cuda()
                i = i + img.size()[0]
                rtime_print('{}/{}'.format(i, len(self.valid_Generator)))
                
                nuclei, contourH, contourRGB = self.net(img, H)
                loss_nuclei = data.soft_dice(nuclei, ground_truth)

                loss_nucleice = data.soft_truncate_ce_loss(nuclei, ground_truth)
                
                loss_H = data.soft_dice(contourH, edge)
                loss_RGB = data.seg_loss(contourRGB, ground_truth)

                loss: torch.Tensor = 0.3 * loss_nuclei + loss_nucleice + 0.3 * loss_H + 0.3 * loss_RGB
                Loss = Loss + loss.item()

        return Loss / len(self.valid_Generator)
    
    def epoch_dice(self) -> tuple[float, float]:
        DICE: list[float] = []
        with torch.no_grad():
            for data_loader in (self.train_loader, self.valid_loader):
                i: int = 0
                nuclei_list = []
                ground_truth_list = []
                for img, H, ground_truth, _ in data_loader:
                    H = H.cuda()
                    img = img.cuda()
                    i = i + img.size()[0]
                    rtime_print('{}/{}'.format(i, len(data_loader)))
                    
                    nuclei, _, _ = self.net(img, H)
                    nuclei = torch.squeeze(nuclei.cpu())
                    
                    nuclei_list.append(nuclei)
                    ground_truth_list.append(torch.squeeze(ground_truth.cpu()))
                    
                DICE.append(dice_evaluation(nuclei_list, ground_truth_list, self.conf.cutoff))

        return (DICE[0], DICE[1])


    def training(self):
        wandb.init(project="segmentation", config={
            "epochs": self.conf.epoches,
            "batch_size": self.conf.batch_size,
            "learning_rate": self.conf.optim_conf['learning_rate'],
            "weight_decay": self.conf.optim_conf['weight_decay'],
            "cutoff": self.conf.cutoff,
        })
        patience: int = 5
        no_improve_epochs: int = 0
        best_val_loss: float = float('inf')
        best_model_state = self.net.state_dict().copy()
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

        for epoch in range(self.conf.epoches):
            my_print('Epoch {}/{}'.format(epoch + 1, self.conf.epoches), '-' * 60)
            
            start = time.time()
            train_loss = self.epoch_train()
            self.scheduler.step()

            val_loss = self.epoch_valid()
            train_dice, val_dice = self.epoch_dice()
            
            end = time.time()
            if val_loss < best_val_loss:
                no_improve_epochs = 0
                best_val_loss = val_loss
                best_model_state = self.net.state_dict().copy()
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs (patience: {patience}). Best validation loss: {best_val_loss:.4f}.\n")
                self.net.load_state_dict(best_model_state)
                break

            history["val_loss"].append(val_loss)
            history['val_dice'].append(val_dice)
            history["train_loss"].append(train_loss)
            history['train_dice'].append(train_dice)
            wandb.log({"train_loss": train_loss, "train_dice": train_dice, "val_loss": val_loss, "val_dice": val_dice})
            my_print('Epoch{} Train Loss:{:.4} Validation Loss:{:.4} Train DICE:[{:.4}] Validation DICE:[{:.4}]  cost time:{:.4}'.format(
                epoch+1, train_loss, val_loss, train_dice, val_dice, str(end-start)))

        torch.save(self.net.state_dict(), self.conf.model_path.format(self.conf.epoches))
        rtime_print('Save AJI model', end='\n')

        history_plot(history)
        wandb.finish()