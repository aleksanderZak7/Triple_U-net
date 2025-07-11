import cv2
import shutil
import numpy as np
from PIL import Image
import skimage.io as io
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split

import Config
from train_test import train_model, test_model
from utils import my_print, rtime_print, get_time


class CrossValidation:
    
    __slots__ = ("_cv", "_conf", "_random_seed", "_val_size", "_fold_dir",
                 "TP_", "PQ_", "IOU_", "AJI_", "DICE_")

    def __init__(self, cv: int, conf: Config.config, validation_size: float = 0.1, random_seed: int = 42) -> None:
        if cv < 2:
            raise ValueError("Cross-validation requires at least 2 folds.")
        if not (0 < validation_size < 1):
            raise ValueError("Validation size must be between 0 and 1.")

        self._cv = cv
        self._conf = conf
        self._random_seed = random_seed
        self._val_size = validation_size
        self._fold_dir = Path(self._conf.input_dir) / "folds"

        self.TP_: list[float] = []
        self.PQ_: list[float] = []
        self.IOU_: list[float] = []
        self.AJI_: list[float] = []
        self.DICE_: list[float] = []

        self._prepare_folds()

    def __del__(self) -> None:
        self.cleanup()

    def __str__(self) -> str:
        return (f"CrossValidation(cv={self._cv}, estimations:"
                f"\nTP: {self.TP_},\nPQ: {self.PQ_},\nIOU: {self.IOU_},"
                f"\nAJI: {self.AJI_},\nDICE:{self.DICE_})")

    def run(self) -> None:
        now: str = get_time(complete=True).replace(':', '-').replace(' ', '_')
        plot_dir: Path = Path(self._conf.save_path) / f"train_plots_{now}"
        plot_dir.mkdir(parents=True, exist_ok=True)
            
        for i in range(self._cv):
            my_print(f"Cross-validation {i + 1}/{self._cv}")

            test_fold = self._fold_dir / f"fold{i + 1}"
            train_folds = [self._fold_dir / f"fold{j + 1}" for j in range(self._cv) if j != i]

            self._test_data_preparation(test_fold)
            self._train_val_data_preparation(train_folds)
            self._prepare_and_save_edge_masks()

            train = train_model(self._conf, str(plot_dir / f"train_plot{i + 1}.png"))
            train.training()

            del train
            test = test_model(self._conf)
            test.test()

            self.TP_.append(test.TP_)
            self.PQ_.append(test.PQ_)
            self.IOU_.append(test.IOU_)
            self.AJI_.append(test.AJI_)
            self.DICE_.append(test.DICE_)

    def cleanup(self) -> None:
        if self._fold_dir.exists():
            shutil.rmtree(self._fold_dir)

    def _prepare_folds(self) -> None:
        mask_dir = Path(self._conf.input_dir) / "Masks"
        original_dir = Path(self._conf.input_dir) / "Original"

        all_masks = sorted(mask_dir.glob('*'))
        all_images = sorted(original_dir.glob('*'))
        assert len(all_images) == len(all_masks), "Mismatch between images and masks count"

        data = list(zip(all_images, all_masks))
        kf = KFold(n_splits=self._cv, shuffle=True, random_state=self._random_seed)

        if self._fold_dir.exists():
            shutil.rmtree(self._fold_dir)
        self._fold_dir.mkdir(parents=True)

        for fold_idx, (_, test_idx) in enumerate(kf.split(data)):  # type: ignore
            fold_path = self._fold_dir / f"fold{fold_idx + 1}"
            (fold_path / "Original").mkdir(parents=True, exist_ok=True)
            (fold_path / "Masks").mkdir(parents=True, exist_ok=True)

            for idx in test_idx:
                img_path, mask_path = data[idx]
                shutil.copy(img_path, fold_path / "Original" / img_path.name)
                shutil.copy(mask_path, fold_path / "Masks" / mask_path.name)

    def _test_data_preparation(self, test_fold: Path) -> None:
        test_mask: Path = test_fold / "Masks"
        test_img: Path = test_fold / "Original"

        test_dir = Path(self._conf.test_data_path)
        label_dir = Path(self._conf.label_path)

        for path in (test_dir, label_dir):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)

        for img_path in test_img.glob('*'):
            shutil.copy(img_path, test_dir / img_path.name)
        for mask_path in test_mask.glob('*'):
            shutil.copy(mask_path, label_dir / mask_path.name)

    def _train_val_data_preparation(self, train_folds: list[Path]) -> None:
        all_masks: list[Path] = []
        all_images: list[Path] = []
        for fold in train_folds:
            all_images.extend(sorted((fold / "Original").glob('*')))
            all_masks.extend(sorted((fold / "Masks").glob('*')))

        data = sorted(zip(all_images, all_masks), key=lambda x: x[0].name)
        imgs, masks = zip(*data)

        del all_masks
        del all_images
        self._prepare_cropped_data(zip(imgs, masks))

    def _read_images_and_masks(self, train_data: zip) -> tuple[list[np.ndarray], list[np.ndarray]]:
        masks: list[np.ndarray] = []
        images: list[np.ndarray] = []

        for image_path, mask_path in train_data:
            image: Image.Image = Image.open(image_path).convert('RGB')
            image_np: np.ndarray = np.asarray(image, dtype=np.uint8)

            mask: Image.Image = Image.open(mask_path).convert('L')
            mask_np: np.ndarray = np.asarray(mask, dtype=np.uint8)
            mask_np = (mask_np >= 127).astype(np.uint8)  # Binarize the mask

            masks.append(mask_np)
            images.append(image_np)

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

    def _shatter_image_and_mask(self, image: np.ndarray, mask: np.ndarray, pad: bool = False,
                                target_size: tuple[int, int] = (60, 60)) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
        
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            images, masks, test_size=self._val_size, random_state=self._random_seed)
        
        my_print(f"Training images: {len(train_imgs)}, Total masks: {len(train_masks)}")
        my_print(f"Validation images: {len(val_imgs)}, Validation masks: {len(val_masks)}")
        
        masks_dir = Path(self._conf.label_path)
        
        for data_type, images, masks in zip(["train", "val"], [train_imgs, val_imgs], [train_masks, val_masks]):
            target_dir: Path = Path(self._conf.train_data_path) if data_type == "train" else Path(self._conf.valid_data_path)

            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True)

            for i, (image_np, mask_np) in enumerate(zip(images, masks)):
                img_pil = Image.fromarray(image_np, 'RGB')
                mask_pil = Image.fromarray(mask_np * 255, 'L')

                file_name = f"{data_type}_item_{i:05d}.png"
                img_pil.save(target_dir / file_name)
                mask_pil.save(masks_dir / file_name)

                rtime_print(f"Saved {i + 1}/{len(images)} images")

    def _prepare_cropped_data(self, img_paths: zip) -> None:
        images, masks = self._read_images_and_masks(img_paths)

        del img_paths
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

        del cropped_masks
        del cropped_images
        total_masks: int = len(masks)
        total_images: int = len(images)
        my_print(f"Total images: {total_images}, Total masks: {total_masks}")
        assert total_masks == total_images, "Mismatch between number of images and masks after shattering."

        self._save_shattered_data(images, masks)

    def _generate_edge_mask(self, mask_image, thickness=1) -> np.ndarray:
        binary_mask = np.uint8(mask_image > 0) * 255

        if np.count_nonzero(binary_mask) == 0:
            return np.zeros_like(binary_mask, dtype=np.float32)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(binary_mask, dtype=np.float32)
        cv2.drawContours(edge_mask, contours, -1, (1,), thickness)

        return edge_mask

    def _prepare_and_save_edge_masks(self) -> None:
        label_dir: Path = Path(self._conf.label_path)
        edge_mask_dir: Path = Path(self._conf.edg_path)

        if edge_mask_dir.exists():
            shutil.rmtree(edge_mask_dir)
        edge_mask_dir.mkdir(parents=True)

        amount_of_masks: int = len(list(label_dir.glob('*')))
        for i, mask_path_obj in enumerate(label_dir.glob('*')):
            full_mask = io.imread(str(mask_path_obj))
            edge_mask = self._generate_edge_mask(full_mask, thickness=1)

            target_edge_path = edge_mask_dir / mask_path_obj.name
            cv2.imwrite(str(target_edge_path), np.ascontiguousarray(np.uint8(edge_mask * 255)))

            rtime_print(f"{i + 1}/{amount_of_masks} edge masks generated.")