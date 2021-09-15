"""Model training."""

import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import pytz
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

import src.metrics as metrics
import src.results as results
import src.transforms as transforms
from src.config import YAMLConfig
from src.cross_validate import make_folds
from src.dataset import CustomDataLoader, CustomDataset
from src.loss import LabelSmoothingLoss
from src.model import CustomModel
from src.oof import get_oof_acc, get_oof_roc
from src.scheduler import GradualWarmupSchedulerV2
from src.utils import seed_all, seed_worker
from src.train import Trainer


def train_on_fold(df_folds: pd.DataFrame, config, fold: int):
    """Train the model on the given fold."""
    model = CustomModel(
        config,
        pretrained=True,
        load_weight=True,
        load_url=False,
        out_dim_heads=[3, 4, 3, 1],
    )

    model.to(config.device)

    ###
    # augmentations_class = getattr(transforms, config.augmentations_class)

    # transforms_train = augmentations_class.from_config(
    #     config.augmentations_train[config.augmentations_class]
    # )
    # transforms_val = augmentations_class.from_config(
    #     config.augmentations_val[config.augmentations_class]
    # )
    ###

    transforms_train, transforms_val = transforms.RANZCR_AUG(image_size=config.image_size)
    train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

    data_dict = {
        "dataset_train_dict": {
            "df": train_df,
            "transforms": transforms_train,
            "transform_norm": True,
            "meta_features": None,
            "mode": "train",
        },
        "dataset_val_dict": {
            "df": val_df,
            "transforms": transforms_val,
            "transform_norm": True,
            "meta_features": None,
            "mode": "train",
        },
        "dataloader_train_dict": {
            "batch_size": config.train_batch_size,
            "shuffle": True,
            "num_workers": config.num_workers,
            "worker_init_fn": seed_worker,
            "pin_memory": True,
        },
        "dataloader_val_dict": {
            "batch_size": config.val_batch_size,
            "shuffle": False,
            "num_workers": config.num_workers,
            "worker_init_fn": seed_worker,
            "pin_memory": True,
        },
    }

    dataloader_dict = CustomDataLoader(config=config, data_dict=data_dict).get_loaders()

    train_loader, val_loader = dataloader_dict["Train"], dataloader_dict["Validation"]

    hongnan_classifier = Trainer(model=model, config=config)
    curr_fold_best_checkpoint = hongnan_classifier.fit(train_loader, val_loader, fold)
    val_df[[str(c) for c in range(config.num_classes)]] = curr_fold_best_checkpoint["oof_preds"]

    return val_df


def train_loop(df_folds: pd.DataFrame, config, fold_num: int = None, train_one_fold=False):
    """Perform the training loop on all folds. Here The CV score is the average of the validation fold metric.
    While the OOF score is the aggregation of all validation folds."""

    cv_score_list = []
    oof_df = pd.DataFrame()
    if train_one_fold:
        _oof_df = train_on_fold(df_folds=df_folds, config=config, fold=fold_num)
    else:
        """The below for loop code guarantees fold starts from 1 and not 0. https://stackoverflow.com/questions/33282444/pythonic-way-to-iterate-through-a-range-starting-at-1"""
        for fold in (number + 1 for number in range(config.num_folds)):
            _oof_df = train_on_fold(df_folds=df_folds, config=config, fold=fold)
            oof_df = pd.concat([oof_df, _oof_df])
            curr_fold_best_score_dict, curr_fold_best_score = get_oof_roc(config, _oof_df)
            cv_score_list.append(curr_fold_best_score)
            print("\n\n\nOOF Score for Fold {}: {}\n\n\n".format(fold, curr_fold_best_score))

        print("CV score", np.mean(cv_score_list))
        print("Variance", np.var(cv_score_list))
        print("Five Folds OOF", get_oof_roc(config, oof_df))
        oof_df.to_csv(os.path.join(config.paths["save_path"], "oof.csv"))


if __name__ == "__main__":

    colab = False
    CURRENT_MODEL = "resnet200d"
    MODELS = {"resnet200d": "/content/reighns/config_RANZCR_resnet200d.yaml"}
    if colab is True:
        if not os.path.exists("/content/reighns"):
            print("new save folder created")
            os.makedirs("/content/reighns")

        yaml_config = YAMLConfig(MODELS[CURRENT_MODEL])

    else:
        yaml_config = YAMLConfig("./config/config_RANZCR_resnet200d.yaml")

    if CURRENT_MODEL == "resnet200d":
        # print(
        #     "We are training on the {} dataset! Please check if you have used BCE LOSS and changed to SIGMOID!".format(
        #         MODELS[CURRENT_MODEL]
        #     )
        # )
        seed_all(seed=yaml_config.seed)
        train_csv = pd.read_csv(yaml_config.paths["csv_path"])

        df_folds = make_folds(train_csv, yaml_config)
        if yaml_config.debug:
            df_folds = df_folds.sample(frac=0.2)

            # yaml_config.train_batch_size = 4
            # yaml_config.val_batch_size = 8
            train_all_folds = train_loop(
                df_folds=df_folds, config=yaml_config, fold_num=1, train_one_fold=True
            )
        else:
            train_all_folds = train_loop(
                df_folds=df_folds, config=yaml_config, fold_num=2, train_one_fold=True
            )  # UNCOMMENT TO TRAIN ALL FOLDS: train_loop(df_folds=df_folds, config=yaml_config)
