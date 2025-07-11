import os
import cv2
import csv
import time
import torch
import random
import numpy as np
import skimage.morphology as sm
from matplotlib import pyplot as plt

SQUARE_KERNEL_KEYWORD = 'square_conv.weight'
last_color = 31


def create_optimizer(conf, model) -> torch.optim.AdamW:
    optimizer_config = conf.optim_conf
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    betas = optimizer_config['betas']
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                         model.parameters()), betas=betas,
                                  lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(conf, optimizer) -> torch.optim.lr_scheduler.ExponentialLR:
    lr_scheduler = conf.lr_scheduler
    gamma = lr_scheduler['gamma']
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)


def history_plot(training_history: dict[str, list[float]], save_path: str | None = None) -> None:
    plt.figure(figsize=(12, 4))
    metrics: tuple[str, str] = ("loss", "dice")

    for idx, key in enumerate(metrics, start=1):
        plt.subplot(1, 2, idx)
        plt.plot(training_history[f"train_{key}"], label="Train")
        plt.plot(training_history[f"val_{key}"], label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel(key.capitalize())
        plt.title(f"{key.capitalize()} History")
        plt.legend()

    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path)
        my_print(f"Plot saved to {save_path}")
    else: 
        plt.show()

def rtime_print(str, end='\r') -> None:
    print('\033[5;{};40m{}\033[0m'.format(
        random.randint(31, 37), str), end=end, flush=True)


def note_by_split(num, split) -> None:
    if num == 0:
        return
    if num % split == 0:
        my_print('handling :{}'.format(num))


def get_filename(path, contain_dir=False, abspath=False, num_only=False, no_num=False) -> tuple[list[str], int] | list[str] | int:
    if not os.path.exists(path):
        my_error('{} not exit!!!'.format(path))
        return []
    FileNames = os.listdir(path)
    ret = []
    num = 0
    for i in range(len(FileNames)):
        f = os.path.join(os.path.join(path, FileNames[i]))
        if contain_dir == 0:
            if os.path.isdir(f):
                continue
        if not abspath:
            f = FileNames[i]
        ret.append(f)
        num += 1
    if num_only:
        return num
    if no_num:
        return ret
    return ret, num


def get_time(complete=False) -> str:
    if not complete:
        return time.strftime("%Y-%m-%d", time.localtime(time.time()))
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def write_csv(content, filename, ifini=0) -> None:
    if ifini:
        with open(filename, 'w+')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(content)
        my_print('Write success!')
        return
    with open(filename, 'a+')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(content)

    my_print('Write success!')


def separate_stain(im: np.ndarray) -> np.ndarray:
    if im is None or im.ndim != 3 or im.shape[2] != 3:
        raise ValueError(
            f"Input image must be a color image (H, W, 3), got shape: {im.shape}")

    H = np.array([0.650, 0.704, 0.286])
    E = np.array([0.072, 0.990, 0.105])
    R = np.array([0.268, 0.570, 0.776])
    stain_matrix = [
        H / np.linalg.norm(H), E / np.linalg.norm(E), R / np.linalg.norm(R)]
    stain_matrix = np.array(stain_matrix)

    try:
        inv_matrix = np.linalg.inv(stain_matrix)
    except np.linalg.LinAlgError as e:
        raise ValueError("Stain matrix is not invertible.") from e

    im = im.astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        im_temp = (-255.0) * np.log((im + 1.0) / 255.0) / np.log(255)

    if np.any(np.isnan(im_temp)) or np.any(np.isinf(im_temp)):
        raise ValueError("Invalid values in log-transformed image.")

    reshaped = im_temp.reshape(-1, 3)
    separated = np.dot(reshaped, inv_matrix)
    image_out = separated.reshape(im.shape)

    with np.errstate(over='ignore', invalid='ignore'):
        image_out = np.exp((255.0 - image_out) * np.log(255.0) / 255.0)

    image_out = np.clip(image_out, 0, 255)

    if np.any(np.isnan(image_out)) or np.any(np.isinf(image_out)):
        raise ValueError("Invalid values in final HE image.")

    return np.uint8(image_out) # type: ignore


def com_str(str, rc=True, sep=' ', last=False) -> str:
    global last_color
    if rc:
        if last:
            last_color = last_color
        else:
            last_color = random.randint(31, 37)
        return '\033[1;{}m{}{}\033[0m'.format(last_color, str, sep)
    else:
        return '\033[1;36m{}{}\033[0m'.format(str, sep)


def my_print(*args, rc=True, sep=' ', if_last=False) -> None:
    for i in range(len(args)-1):
        if i == 0:
            print(com_str(args[i], rc, '', last=if_last), end='')
            continue
        print(com_str(args[i], rc, sep, last=if_last), end='')
    print(com_str(args[len(args)-1], rc, sep, last=if_last))


def my_error(str) -> None:
    print('\033[1;31m{}\033[0m'.format(str))


def adjustData(img, mask) -> tuple[np.ndarray, np.ndarray]:
    img = img / 255
    mask = (mask > 200)*1
    return (img, mask)


def get_attention(img) -> np.ndarray:
    sep = separate_stain(img)
    sep = np.reshape((sep[:, :, 0] < 230), [
                     np.shape(img)[0], np.shape(img)[1]])
    # remove_small_objects can only handle bool type image.
    sep = sm.remove_small_objects(sep, min_size=100, connectivity=2)
    kernel = sm.disk(1)
    sep = sm.dilation(sep, kernel)
    sep = imfill(sep)
    return sep


def my_load(model, hdf5) -> dict[str, torch.Tensor]:
    ret = {}
    if isinstance(hdf5, str):
        ud = torch.load(hdf5)
    else:
        ud = hdf5
    for i in model.state_dict().keys():
        for h in ud.keys():
            if i in h or h in i:
                # print(i,h)
                ret[i] = ud[h]
    return ret


def imfill(im_in) -> np.ndarray:
    if im_in.ndim != 2:
        my_error('Only handle Binary but get image dim:{}!'.format(im_in.ndim))
        return im_in
    if np.max(im_in) > 1:
        im_th = im_in
    else:
        im_th = im_in*255
    im_th = im_th.astype(np.uint8)
    h, w = im_in.shape[:2]
    temp = np.zeros((h+2, w+2), np.uint8)
    temp[1:h+1, 1:w+1] = im_in
    mask = np.zeros((h+4, w+4), np.uint8)
    cv2.floodFill(temp, mask, (0, 0), 255, cv2.FLOODFILL_FIXED_RANGE) # type: ignore
    im_floodfill_inv = ~temp[1:h+1, 1:w+1]
    return (im_floodfill_inv > 1)*1


def _fuse_kernel(kernel, gamma, std) -> torch.Tensor:
    b_gamma = torch.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = b_gamma.repeat(
        1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    b_std = torch.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = b_std.repeat(1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    return kernel * b_gamma / b_std


def _add_to_square_kernel(square_kernel, asym_kernel) -> None:
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    square_h = square_kernel.shape[2]
    square_w = square_kernel.shape[3]
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                  square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel


def convert_acnet_weights(hdf5, eps=1e-5) -> dict[str, torch.Tensor]:
    train_dict = torch.load(hdf5)

    deploy_dict = {}
    square_conv_var_names = [
        name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for square_name in square_conv_var_names:
        square_kernel = train_dict[square_name]
        square_mean = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'square_bn.running_mean')]
        square_std = torch.sqrt(train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'square_bn.running_var')] + eps)
        square_gamma = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'square_bn.weight')]
        square_beta = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'square_bn.bias')]

        ver_kernel = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'ver_conv.weight')]
        ver_mean = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'ver_bn.running_mean')]
        ver_std = torch.sqrt(train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'ver_bn.running_var')] + eps)
        ver_gamma = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'ver_bn.weight')]
        ver_beta = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'ver_bn.bias')]

        hor_kernel = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'hor_conv.weight')]
        hor_mean = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'hor_bn.running_mean')]
        hor_std = torch.sqrt(train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'hor_bn.running_var')] + eps)
        hor_gamma = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'hor_bn.weight')]
        hor_beta = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'hor_bn.bias')]

        fused_bias = square_beta + ver_beta + hor_beta - square_mean * square_gamma / square_std \
            - ver_mean * ver_gamma / ver_std - hor_mean * hor_gamma / hor_std
        fused_kernel = _fuse_kernel(square_kernel, square_gamma, square_std)
        _add_to_square_kernel(fused_kernel, _fuse_kernel(
            ver_kernel, ver_gamma, ver_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(
            hor_kernel, hor_gamma, hor_std))

        deploy_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        deploy_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias

    for k, v in train_dict.items():
        if 'hor_' not in k and 'ver_' not in k and 'square_' not in k:
            deploy_dict[k] = v
    # print(deploy_dict.keys())
    return deploy_dict


def convert_no_norm_acnet_weights(hdf5, eps=1e-5) -> dict[str, torch.Tensor]:
    train_dict = torch.load(hdf5)

    deploy_dict = {}
    square_conv_var_names = [
        name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for square_name in square_conv_var_names:
        square_kernel = train_dict[square_name]
        ver_kernel = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'ver_conv.weight')]
        hor_kernel = train_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'hor_conv.weight')]

        _add_to_square_kernel(square_kernel, ver_kernel)
        _add_to_square_kernel(square_kernel, hor_kernel)

        deploy_dict[square_name.replace(
            SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = square_kernel

    for k, v in train_dict.items():
        if 'hor_' not in k and 'ver_' not in k and 'square_' not in k:
            deploy_dict[k] = v
    # print(deploy_dict.keys())
    return deploy_dict
