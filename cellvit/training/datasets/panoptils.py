# -*- coding: utf-8 -*-
# Panoptils Dataset
#
# Dataset information: https://sites.google.com/view/panoptils/home
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import csv
from pathlib import Path
from typing import Callable, Union, Tuple, List

import albumentations as A
import numpy as np
import torch
import torchstain
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class PanoptilsDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str,
        filelist_path: Union[Path, str] = None,
        transforms: Callable = A.Compose(
            [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
        ),
        normalize_stains: bool = False,
    ) -> None:
        """Panoptils Dataset for Cell Segmentation

        Args:
            dataset_path (Union[Path, str]): Path to the dataset parent folder
            split (str): Split of the dataset (train, test)
            filelist_path (Union[Path, str], optional): Path to a filelist (csv) to retrieve just a subset of images to use.
                Otherwise, all images from split are used. Defaults to None.
            transforms (Callable, optional): Transformations. Defaults to A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]).
            normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        """
        super().__init__()
        self.transforms = transforms
        self.normalize_stains = normalize_stains
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = self.dataset_path / self.split / "images"
        self.annotation_path = self.dataset_path / self.split / "annotations"

        self.images = [f for f in sorted(self.image_path.glob("*.png"))]

        if filelist_path is not None:
            selected_files = []
            with open(filelist_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    selected_files.append(row[0])
            self.images = [f for f in self.images if f.stem in selected_files]

        self.annotations = []
        for img_path in self.images:
            img_name = img_path.stem
            self.annotations.append(self.annotation_path / f"{img_name}.csv")

        self.cache_images = {}
        self.cache_annotations = {}
        self.label_map = {
            0: "Other Cells",
            1: "Epithelial Cells",
            2: "Stromal Cells",
            3: "TILs",
        }

    def cache_dataset(self) -> None:
        logger.warning("No cache available due to dataset size")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, list, list, str]:
        """Get item from dataset

        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, list, list, str]:
            * Image
            * List of detections
            * List of types
            * Name of the Patch
        """
        img_path = self.images[index]
        img_name = img_path.stem
        img = Image.open(img_path)
        img = img.convert("RGB")

        annot_path = self.annotations[index]
        with open(annot_path, "r") as file:
            reader = csv.reader(file)
            cell_annot = list(reader)
        detections = [(int(x), int(y)) for x, y, _ in cell_annot]
        types = [int(int(t)) for _, _, t in cell_annot]

        if self.normalize_stains:
            img = to_tensor(img)
            img = (255 * img).type(torch.uint8)
            img, _, _ = self.normalizer.normalize(img)
            img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))

        img = np.array(img).astype(np.uint8)

        if self.transforms:
            transformed = self.transforms(image=img, keypoints=detections)
            img = transformed["image"]
            detections = transformed["keypoints"]
            types = [types[idx] for idx, _ in enumerate(detections)]

        return img, detections, types, img_name

    @staticmethod
    def collate_batch(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, List[list], List[list], List[str]]:
        """Create a custom batch

        Needed to unpack List of tuples with dictionaries and array

        Args:
            batch (List[Tuple]): Input batch consisting of a list of tuples (patch, cell_coordinates, cell_types, patch_names)

        Returns:
            Tuple[torch.Tensor, List[list], List[list], List[str]]:
                * patches with shape [batch_size, 3, patch_size, patch_size]
                * List of detections, each entry is a list with one entry for each ground truth cell
                * list of types, each entry is the cell type for each ground truth cell
                * list of patch names
        """
        imgs, detections_list, types_list, names = zip(*batch)
        imgs = torch.stack(imgs)
        return imgs, list(detections_list), list(types_list), list(names)
