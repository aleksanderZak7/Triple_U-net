import random
import numpy as np

last_color = 31


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