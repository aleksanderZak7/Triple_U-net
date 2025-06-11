import wandb    # !

import os
import time
import data
import torch
import metrics
import skimage
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt

import model
import Config
import transform
from utils import *


def generate_plot(training_history: dict[str, list[float]]):
    plt.plot(training_history['train_loss'], label='Train', marker='o')
    plt.plot(training_history['val_loss'], label='Validation', marker='s')

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss')

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def _create_optimizer(conf, model):
    optimizer_config = conf.optim_conf
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    betas = optimizer_config['betas']
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                         model.parameters()), betas=betas,
                                  lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(conf, optimizer):
    lr_scheduler = conf.lr_scheduler
    gamma = lr_scheduler['gamma']
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)


def evaluation(pre, mask, cutoff, min_size=10):
    IOU = np.array([])
    DICE = np.array([])
    AJI = np.array([])
    TP = np.array([])
    PQ = np.array([])

    for i in range(len(pre)):
        img = skimage.morphology.remove_small_objects(
            np.array(pre[i]) > cutoff, min_size=min_size)
        PQ = np.append(PQ, metrics.get_fast_pq(
            np.array(mask[i], dtype='uint8'), np.array(img, dtype='uint8'))[0][2])
        IOU = np.append(IOU, metrics.compute_iou(img, mask[i], cutoff))
        DICE = np.append(DICE, metrics.compute_F1(
            img, mask[i], cutoff))   # Dice -> compute_F1
        AJI = np.append(AJI, metrics.get_fast_aji(mask[i], img))
        TP = np.append(TP, metrics.compute_TP_ratio(img, mask[i], cutoff))

    my_print('Num is:{} '.format(len(PQ)), 'cutoff=[{}]'.format(cutoff), 'PQ=[{:.6}]'.format(np.mean(PQ)),
             'DICE=[{:.6}]'.format(np.mean(DICE)), 'AJI=[{:.6}]'.format(np.mean(AJI)))

    return np.mean(PQ), np.mean(DICE), np.mean(AJI)


class test_model(object):
    def __init__(self, conf):
        super(test_model, self).__init__()
        self.conf = conf
        self.net = torch.load(conf.model_path)
        self.datagen = data.trainGenerator(
            conf.test_data_path, conf.label_path, conf.edg_path)

    def test(self):
        rgb_pre = []
        file_ = []
        HE_pre = []
        nuclei_pre = []
        grounf_truth = []
        file_ = []
        for index in range(len(self.datagen)):
            rtime_print('{}/{}'.format(index+1, len(self.datagen)))
            img, file = self.datagen.load_image(index)
            if self.datagen.just_img_name(file):
                continue
            # kanał HE barwienie hematoksylina-eozyna -
            HE = self.datagen.load_HE(index)
            img = torch.unsqueeze(img.cuda(), 0)
            HE = torch.unsqueeze(HE.cuda(), 0)  # type: ignore

            mask = self.datagen.load_mask(index)
            nuclei, outh, outrgb = self.predition(img, HE, file)
            rgb_pre.append(torch.squeeze(outrgb.cpu()))
            nuclei_pre.append(nuclei)
            HE_pre.append(torch.squeeze(outh.cpu()))
            grounf_truth.append(torch.squeeze(mask))
            file_.append(file)

        evaluation(nuclei_pre, grounf_truth, self.conf.cutoff)

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


class train(object):
    def __init__(self, conf):
        super(train, self).__init__()
        self.conf = conf
        self.net = model.net().cuda()

        self.optimizer = _create_optimizer(conf, self.net)
        self.scheduler = _create_lr_scheduler(conf, self.optimizer)

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

    def epoch_train(self):
        i: int = 0
        Loss: float = 0.0
        self.net.train()

        for d in self.train_loader:
            img, H, ground_truth, edge = d
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
            Loss = Loss+loss.item()
        return Loss / len(self.train_data_Generator)

    def epoch_valid(self):
        i: int = 0
        Loss: float = 0.0
        self.net.eval()

        for d in self.valid_loader:
            img, H, ground_truth, edge = d
            H = H.cuda()
            img = img.cuda()
            i = i + img.size()[0]
            rtime_print('{}/{}'.format(i, len(self.valid_Generator)))
            with torch.no_grad():
                nuclei, contourH, contourRGB = self.net(img, H)
                loss_nuclei = data.soft_dice(nuclei, ground_truth)

                loss_nucleice = data.soft_truncate_ce_loss(
                    nuclei, ground_truth)
                loss_H = data.soft_dice(contourH, edge)
                loss_RGB = data.seg_loss(contourRGB, ground_truth)

                loss: torch.Tensor = 0.3 * loss_nuclei + \
                    loss_nucleice + 0.3 * loss_H + 0.3 * loss_RGB
                Loss = Loss + loss.item()

        return Loss/len(self.valid_Generator)

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
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.conf.epoches):
            my_print('Epoch {}/{}'.format(epoch + 1,
                     self.conf.epoches), '-' * 60)
            start = time.time()
            train_loss = self.epoch_train()
            self.scheduler.step()

            val_loss = self.epoch_valid()
            end = time.time()

            if val_loss < best_val_loss:
                no_improve_epochs = 0
                best_val_loss = val_loss
                best_model_state = self.net.state_dict().copy()
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs (patience: {patience}). Best validation loss: {best_val_loss:.4f}.\n")
                self.net.load_state_dict(best_model_state)
                break

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            my_print('Epoch{} Train Loss:{:.6} Validation Loss:{:.6}  cost time:{:.6}'.format(
                epoch, train_loss, val_loss, str(end-start)))

        torch.save(self.net, self.conf.model_path.format(self.conf.epoches))
        rtime_print('Save AJI model', end='\n')

        generate_plot(history)
        wandb.finish()


if __name__ == '__main__':
    epoch: int = 100
    train_model: bool = False

    conf = Config.config()
    conf.epoches = epoch
    conf.train = train_model
    start = time.time()
    end = 0
    if conf.train:
        my_print('data_path:        {}'.format(conf.train_data_path))
        my_print('label_path:       {}'.format(conf.label_path))
        my_print('valid_data_path:  {}'.format(conf.valid_data_path))
        my_print('epoches num:      {}'.format(conf.epoches))

        my_activation = train(conf)
        my_print('Total epoches ({})'.format(conf.epoches))
        my_activation.training()
        end = time.time()
    else:
        my_print('test_data_path: {}'.format(conf.test_data_path))
        my_activation = test_model(conf)
        my_activation.test()
        end = time.time()
    my_print('Running time:{}'.format(str(end-start)))
