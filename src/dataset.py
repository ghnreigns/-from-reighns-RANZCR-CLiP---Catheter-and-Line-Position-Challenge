"""A dataset loader."""
import os
from typing import Optional, List, Dict

import cv2
import numpy as np
import pandas as pd
import torch

from src.utils import check_df_ext, get_file_type


class CustomDataset(torch.utils.data.Dataset):
    """The Custom Dataset. transforms is now an abstract class"""

    def __init__(
        self,
        config: type,
        df: pd.DataFrame = None,
        file_list: List = None,
        transforms: type = None,
        transform_norm: bool = True,
        meta_features: bool = None,
        mode: str = "train",
    ):
        """Construct a Custom dataset."""

        self.df = df
        self.file_list = file_list
        self.config = config
        self.transforms = transforms
        self.transform_norm = transform_norm
        self.meta_features = meta_features
        self.mode = mode

        if self.transforms is None:
            assert self.transform_norm is False
            print("Transforms is None and Transform Normalization is not " "initialized!")

        self.image_extension = get_file_type(
            image_folder_path=config.paths["train_path"], allowed_extensions=None
        )

    def __len__(self):
        """Get the dataset length."""
        return len(self.df) if self.df is not None else len(self.file_list)

    def __getitem__(self, idx: int):
        """Get a row from the dataset."""

        image_id = self.df[self.config.image_col_name].values[idx]
        label = torch.zeros(1)

        if self.mode == "train":
            label = self.df[self.config.class_col_name].values[idx]
            label = torch.as_tensor(data=label, dtype=torch.int64, device=None)
            image_path = os.path.join(
                self.config.paths["train_path"],
                "{}{}".format(image_id, self.image_extension),
            )

        else:
            image_path = os.path.join(
                self.config.paths["test_path"],
                "{}{}".format(image_id, self.image_extension),
            )

        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Specific for this competition, preprocess to remove black borders

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = image > 0
        image = image[np.ix_(mask.any(1), mask.any(0))]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if not self.transform_norm:
            image = image.astype(np.float32) / 255.0

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
            # image = self.transforms.augment(image)
        else:
            image = torch.as_tensor(data=image, dtype=torch.float32, device=None)

        if self.meta_features is not None:
            meta = np.array(self.df.iloc[idx][self.meta_features].values, dtype=np.float32)
            return image_id, (image, meta), label

        # Note this is important if you use BCE loss. Must make labels to float for some reason
        if self.config.criterion_train == "BCEWithLogitsLoss":
            label = torch.as_tensor(data=label, dtype=torch.float32, device=None)

        ### We using PyTorch so channels first "Raise Channels Last Error"
        if image.shape[0] != 3 and image.shape[0] != 1:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return image_id, image, label


class CustomDataLoader:

    """
    Class which will return dictionary containing dataloaders for training and validation.

    Arguments:

        config : A dictionary which contains the following keys and corresponding values.
        Keys and Values of config
            train_paths : list of paths of images for training.
            valid_paths : list of paths of images for validation.
            train_targets : targets for training.
            valid_targets : targets for validation.
            train_augmentations : Albumentations augmentations to use while training.
            valid_augmentations : Albumentations augmentations to use while validation.

        Reason why using dictionary ? -> It will keep all of the data pipeline clean and simple.

    Return :
            Dictionary containing training dataloaders and validation dataloaders
    """

    def __init__(self, config: type, data_dict: Dict):
        self.config = config
        self.data_dict = data_dict
        self.train_dataset = CustomDataset(self.config, **self.data_dict["dataset_train_dict"])
        self.valid_dataset = CustomDataset(self.config, **self.data_dict["dataset_val_dict"])

    def get_loaders(self):

        """
        Function which will return dictionary of dataloaders
        Arguments:

            train_bs : Batch Size for train loader.
            valid_bs : Batch Size for valid loader.
            num_workers : num_workers to be used by dataloader.
            drop_last : whether to drop last batch or not.
            shuffle : whether to shuffle inputs
            sampler : if dataloader is going to use a custom sampler pass the sampler argument.

        Returns :

            Dictionary with Training and Validation Loaders.

        """

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, **self.data_dict["dataloader_train_dict"]
        )

        val_loader = torch.utils.data.DataLoader(
            self.valid_dataset, **self.data_dict["dataloader_val_dict"]
        )

        dataloader_dict = {"Train": train_loader, "Validation": val_loader}

        return dataloader_dict
