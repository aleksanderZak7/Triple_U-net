import random
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split

import Config
from train_test import train_model
from utils import my_print, rtime_print


class CrossValidation:
    def __init__(self, cv: int, conf: Config.config, validation_size: float = 0.1, random_seed: int = 42) -> None:
        self._cv = cv
        self._conf = conf
        self._random_seed = random_seed
        self._val_size = validation_size
        self._temp_dir = Path(self._conf.input_dir) / "temp"

        self.TP_: list[float] = []
        self.PQ_: list[float] = []
        self.IOU_: list[float] = []
        self.AJI_: list[float] = []
        self.DICE_: list[float] = []

        self._prepare_folds()

    def __str__(self) -> str:
        return (f"CrossValidation(cv={self._cv}, estimations:"
                f"\nTP: {self.TP_},\nPQ: {self.PQ_},\nIOU: {self.IOU_},"
                f"\nAJI: {self.AJI_},\nDICE:{self.DICE_})")

    def _prepare_folds(self) -> None:
        mask_dir = Path(self._conf.input_dir) / "Masks"
        original_dir = Path(self._conf.input_dir) / "Original"

        all_masks = sorted(mask_dir.glob('*'))
        all_images = sorted(original_dir.glob('*'))
        assert len(all_images) == len(all_masks), "Mismatch between images and masks count"

        data = list(zip(all_images, all_masks))
        random.seed(self._random_seed)
        random.shuffle(data)

        kf = KFold(n_splits=self._cv, shuffle=False)
        base_folds = Path(self._conf.input_dir) / "folds"
        if base_folds.exists():
            shutil.rmtree(base_folds)
        base_folds.mkdir(parents=True)

        for fold_idx, (_, test_idx) in enumerate(kf.split(data)):  # type: ignore
            fold_path = base_folds / f"fold{fold_idx + 1}"
            (fold_path / "Original").mkdir(parents=True, exist_ok=True)
            (fold_path / "Masks").mkdir(parents=True, exist_ok=True)

            for idx in test_idx:
                img_path, mask_path = data[idx]
                shutil.copy(img_path, fold_path / "Original" / img_path.name)
                shutil.copy(mask_path, fold_path / "Masks" / mask_path.name)

    def run(self) -> None:
        base_folds = Path(self._conf.input_dir) / "folds"
        for i in range(self._cv):
            my_print(f"Cross-validation {i + 1}/{self._cv}")

            test_fold = base_folds / f"fold{i + 1}"
            train_folds = [base_folds /
                           f"fold{j + 1}" for j in range(self._cv) if j != i]

            self.test_data_split(test_fold)
            self.train_val_data_split(train_folds)
            
            self._prepare_training_data() #TODO: change loading data from temp dir

            train = train_model(self._conf)
            train.training()

    def test_data_split(self, test_fold: Path) -> None:
        test_mask: Path = test_fold / "Masks"
        test_img: Path = test_fold / "Original"

        test_dir = Path(self._conf.test_data_path)
        label_dir = Path(self._conf.label_path)
        
        for path in (test_dir, label_dir):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)

        test_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        for img_path in test_img.glob('*'):
            shutil.copy(img_path, test_dir / img_path.name)
        for mask_path in test_mask.glob('*'):
            shutil.copy(mask_path, label_dir / mask_path.name)

    def train_val_data_split(self, train_folds: list[Path]) -> None:
        all_images = []
        all_masks = []
        for fold in train_folds:
            all_images.extend(sorted((fold / "Original").glob('*')))
            all_masks.extend(sorted((fold / "Masks").glob('*')))

        data = sorted(zip(all_images, all_masks), key=lambda x: x[0].name)
        imgs, masks = zip(*data)

        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            imgs, masks, test_size=self._val_size, random_state=self._random_seed)

        label_dir = Path(self._conf.label_path)
        valid_dir = Path(self._conf.valid_data_path)
        valid_dir.mkdir(parents=True, exist_ok=True)
        
        temp_img_dir: Path = self._temp_dir / "train"
        temp_masks_dir: Path = self._temp_dir / "masks"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        temp_masks_dir.mkdir(parents=True, exist_ok=True)
        
        for path in (valid_dir, temp_img_dir, temp_masks_dir):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)

        for img, mask in zip(val_imgs, val_masks):
            shutil.copy(img, valid_dir / img.name)
            shutil.copy(mask, label_dir / mask.name)

        for img, mask in zip(train_imgs, train_masks):
            shutil.copy(img, temp_img_dir / img.name)
            shutil.copy(mask, temp_masks_dir / mask.name)
        
    def _read_images_and_masks(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        masks: list[np.ndarray] = []
        images: list[np.ndarray] = []

        temp_masks_dir: Path = self._temp_dir / "masks"
        images_directory: Path = self._temp_dir / "train"
        img_list: list[Path] = sorted([p for p in images_directory.glob('*.png')])

        for image_path in img_list:
            image: Image.Image = Image.open(image_path).convert('RGB')
            mask_path: Path = temp_masks_dir / image_path.name

            mask: Image.Image = Image.open(mask_path).convert('L')
            image_np: np.ndarray = np.asarray(image, dtype=np.uint8)

            mask_np: np.ndarray = np.asarray(mask, dtype=np.uint8)
            mask_np = (mask_np >= 127).astype(np.uint8)  # Binarize the mask

            images.append(image_np)
            masks.append(mask_np)
        
        shutil.rmtree(self._temp_dir)
        return images, masks
    
    def _crop_image_to_masked_region(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(mask)
        if len(ys) == 0 or len(xs) == 0:
                return image, mask

        top: int = ys.min()
        bottom: int = ys.max()

        left: int = xs.min()
        right: int = xs.max()

        cropped_image: np.ndarray = image[top:bottom+1, left:right+1, :]
        cropped_mask: np.ndarray = mask[top:bottom+1, left:right+1]

        return cropped_image, cropped_mask

    def _shatter_image_and_mask(self, image: np.ndarray, mask: np.ndarray, 
                               target_size: tuple[int, int] = (60, 60), pad: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
        h: int = image.shape[0]
        w: int = image.shape[1]
        th, tw = target_size

        if pad:
            pad_h: int = (th - h % th) % th
            pad_w: int = (tw - w % tw) % tw
            pad_top: int = pad_h // 2
            pad_bottom: int = pad_h - pad_top
            pad_left: int = pad_w // 2
            pad_right: int = pad_w - pad_left

            mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

            h, w = image.shape[:2]

        result_images: list[np.ndarray] = []
        result_masks: list[np.ndarray] = []

        for y in range(0, h - th + 1, th):
            for x in range(0, w - tw + 1, tw):
                result_images.append(image[y:y+th, x:x+tw, :])
                result_masks.append(mask[y:y+th, x:x+tw])

        return result_images, result_masks
    
    def _save_shattered_data(self, images: list[np.ndarray], masks: list[np.ndarray]) -> None:
        masks_dir = Path(self._conf.label_path)
        image_dir = Path(self._conf.train_data_path)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (image_np, mask_np) in enumerate(zip(images, masks)):
            img_pil = Image.fromarray(image_np, 'RGB')
            mask_pil = Image.fromarray(mask_np * 255, 'L')
            
            file_name = f"train_item_{i:05d}.png"
            img_pil.save(image_dir / file_name)
            mask_pil.save(masks_dir / file_name)
            
            rtime_print(f"Saved {i + 1}/{len(images)} images")

    def _prepare_training_data(self) -> None:
        images, masks = self._read_images_and_masks()
        
        cropped_masks: list[np.ndarray] = []
        cropped_images: list[np.ndarray] = []
        
        for image, mask in zip(images, masks):
            cropped_image, cropped_mask = self._crop_image_to_masked_region(image, mask)
            cropped_images.append(cropped_image)
            cropped_masks.append(cropped_mask)
        
        masks: list[np.ndarray] = []
        images: list[np.ndarray] = []
        for image, mask in zip(cropped_images, cropped_masks):
            shattered_images, shattered_masks = self._shatter_image_and_mask(image, mask)
            images.extend(shattered_images)
            masks.extend(shattered_masks)

        total_masks: int = len(masks)
        total_images: int = len(images)
        my_print(f"Total training images: {total_images}, Total masks: {total_masks}")
        assert total_masks == total_images, "Mismatch between number of images and masks after shattering."
        
        self._save_shattered_data(images, masks)