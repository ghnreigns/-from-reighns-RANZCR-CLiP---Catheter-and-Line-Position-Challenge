"""A module for performing image augmentations."""
from abc import ABC, abstractmethod

import albumentations
import numpy as np
import torch

# import torchtoolbox.transform
import torchvision
from albumentations.pytorch.transforms import ToTensorV2

# from discolight.disco import disco

# from autoaugment.auto_augment import AutoAugment


class Augmentation(ABC):
    """A standard interface for performing augmentations."""

    # The object that contains the augmentations that may be specified
    # in the configuration information. This must be specified by the
    # implementing class.
    augmentations_store = None
    # The function or callable object that is used to construct a composition
    # of augmentations. This must be specified by the implementing class.
    compose_constructor = None

    @abstractmethod
    def augment(self, image):
        """Augment an image."""

    @classmethod
    def from_config(klass, augmentations):
        """Construct an augmentation from configuration data.

        This function takes a list of augmentations in the form

            [ {"name": "Augmentation1", "params": {"param1": 1.0, ...}},
              {"name": "Augmentation2"},
              ...
            ]

        and returns an Augmentation class which will perform a
        composition of the specified augmentations. The name of each
        augmentation corresponds to a member of the
        augmentations_store object. The function or callable object in
        compose_constructor will be used to construct the composition
        augmentation. This composition augmentation will then be
        supplied to the constructor of the Augmentation class to
        return a new class instance.
        """
        augmentation_objs = [
            getattr(klass.augmentations_store, augmentation["name"])(
                **augmentation.get("params", {})
            )
            for augmentation in augmentations
        ]

        return klass(klass.compose_constructor(augmentation_objs))


class AlbumentationsAugmentation(Augmentation):
    class AlbumentationsStore:
        """A wrapper that exposes ToTensorV2 alongside other augmentations."""

        def __getattr__(self, name):

            if name == "ToTensorV2":
                return ToTensorV2

            return getattr(albumentations, name)

    augmentations_store = AlbumentationsStore()
    compose_constructor = albumentations.Compose

    def __init__(self, transforms: albumentations.core.composition.Compose):
        self.transforms = transforms

    def augment(self, image):
        albu_dict = {"image": image}
        transform = self.transforms(**albu_dict)
        return transform["image"]


class TorchTransforms(Augmentation):
    class TorchTransformsStore:
        def __getattr__(self, name):

            if name == "AutoAugment":
                return AutoAugment

            return getattr(torchvision.transforms, name)

    augmentations_store = TorchTransformsStore()
    compose_constructor = torchvision.transforms.Compose

    def __init__(self, transforms: torchvision.transforms.transforms.Compose):
        self.transforms = transforms

    def augment(self, image):
        if isinstance(image, np.ndarray):
            image = torchvision.transforms.ToPILImage()(image)
        transformed_image = self.transforms(image)
        return transformed_image


def RANZCR_AUG(image_size=512):
    transforms_train = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(image_size, image_size, scale=(0.9, 1), p=1),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7
            ),
            albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
            albumentations.OneOf(
                [
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(num_steps=5, distort_limit=1.0),
                    albumentations.ElasticTransform(alpha=3),
                ],
                p=0.2,
            ),
            albumentations.OneOf(
                [
                    albumentations.GaussNoise(var_limit=[10, 50]),
                    albumentations.GaussianBlur(),
                    albumentations.MotionBlur(),
                    albumentations.MedianBlur(),
                ],
                p=0.2,
            ),
            albumentations.Resize(image_size, image_size),
            albumentations.OneOf(
                [
                    albumentations.JpegCompression(),
                    albumentations.Downscale(scale_min=0.1, scale_max=0.15),
                ],
                p=0.2,
            ),
            albumentations.IAAPiecewiseAffine(p=0.2),
            albumentations.IAASharpen(p=0.2),
            albumentations.Cutout(
                max_h_size=int(image_size * 0.1),
                max_w_size=int(image_size * 0.1),
                num_holes=5,
                p=0.5,
            ),
            albumentations.Normalize(mean=[0.4887381077884414], std=[0.23064819430546407]),
        ]
    )

    transforms_valid = albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(mean=[0.4887381077884414], std=[0.23064819430546407]),
        ]
    )
    return transforms_train, transforms_valid
