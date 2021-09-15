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

from src import metrics
from src import results
from src import transforms
from src.config import YAMLConfig
from src.cross_validate import make_folds
from src.dataset import CustomDataLoader, CustomDataset
from src.loss import LabelSmoothingLoss
from src.model import CustomModel
from src.oof import get_oof_acc, get_oof_roc
from src.scheduler import GradualWarmupSchedulerV2
from src.utils import seed_all, seed_worker
from src.transforms import RANZCR_AUG


class Trainer:
    """A class to perform model training."""

    def __init__(self, model, config, early_stopping=None):
        """Construct a Trainer instance."""
        self.model = model
        self.config = config
        self.early_stopping = early_stopping
        self.epoch = 0
        # loss history and monitored metrics history stores each epoch's results, and save it to the weights, so later we can access it easily, we can also access it by calling the attribute
        self.loss_history = []
        self.monitored_metrics_history = []
        self.save_path = config.paths["save_path"]
        if not os.path.exists(self.save_path):
            print("new save folder created")
            os.makedirs(self.save_path)

        self.criterion_train = getattr(torch.nn, config.criterion_train)(
            **config.criterion_params[config.criterion_train]
        )
        self.criterion_val = getattr(torch.nn, config.criterion_val)(
            **config.criterion_params[config.criterion_val]
        )
        self.optimizer = getattr(torch.optim, config.optimizer)(
            self.model.parameters(), **config.optimizer_params[config.optimizer]
        )
        self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler)(
            optimizer=self.optimizer, **config.scheduler_params[config.scheduler]
        )

        # This is built upon self.scheduler, note the params in self.schedule must match number of epochs.
        warmup_epoch = 1
        warmup_factor = 10
        # use initial lr divide by warmup factpr
        self.scheduler_warmup = GradualWarmupSchedulerV2(
            self.optimizer,
            multiplier=10,
            total_epoch=warmup_epoch,
            after_scheduler=self.scheduler,
        )

        """scaler is only used when use_amp is True, use_amp is inside config."""
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.selected_results_val = [
            results.construct_result(result, self.config) for result in self.config.results_val
        ]

        self.validation_results = results.ValidationResults(
            self, self.selected_results_val, self.config
        )

        self.selected_results_train = [
            results.construct_result(result, self.config) for result in self.config.results_train
        ]

        self.training_results = results.TrainingResults(
            self, self.selected_results_train, self.config
        )

        self.best_val_results = {}

        self.saved_val_results = {}
        """https://stackoverflow.com/questions/1398674/display-the-time-in-a-different-time-zone"""
        self.date = datetime.datetime.now(pytz.timezone("Asia/Singapore")).strftime("%Y-%m-%d")

        print(config.device)
        self.log(
            "[Trainer prepared]: We are using {} device with {} worker(s).\nThe monitored metric is {}\n".format(
                self.config.device,
                self.config.num_workers,
                self.config.monitored_result,
            )
        )

        self.log(
            "In results.py, we are dealing with a Multiclass/label problem, using softmax/sigmoid, etc\n"
        )

        self.log("In dataset.py, we are doing some preprocessing for this dataset!\n")
        self.log("We are using warmup scheduler! Please turn off other scheduler steps.")

    def fit(self, train_loader, val_loader, fold: int):
        """Fit the model on the given fold."""
        self.log("Training on Fold {} and using {}".format(fold, self.config.model_name))

        for _epoch in range(self.config.n_epochs):
            # Getting the learning rate after each epoch!
            lr = self.optimizer.param_groups[0]["lr"]
            # Step scheduler.
            """
            Unlike most of the schedulers, which start with the given initial learning rate and adapt it step by step, this gradual warmup scheduler should modify the initial learning rate as zero before any backward update on weights.

            But if you want to avoid this warning message, there is a walk-around. See the latest code. I call 'optimizer.step()' with zero gradients, right after I create schedulers.
            """

            self.scheduler_warmup.step(_epoch)
            ###
            timestamp = datetime.datetime.now(pytz.timezone("Asia/Singapore")).strftime(
                "%Y-%m-%d %H-%M-%S"
            )

            self.log("\n{}\nLR: {}".format(timestamp, lr))

            train_start_time = time.time()

            train_results_computed = self.train_one_epoch(train_loader)

            train_end_time = time.time()

            train_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(train_end_time - train_start_time)
            )

            train_reported_results = [
                result.report(train_results_computed[result.__class__.__name__])
                for result in self.selected_results_train
                if isinstance(result, results.ReportableResult)
            ]

            train_result_str = " | ".join(
                [
                    "Training Epoch: {}".format(self.epoch + 1),
                    *train_reported_results,
                    "Time Elapsed: {}".format(train_elapsed_time),
                ]
            )

            self.log("[TRAIN RESULT]: {}".format(train_result_str))

            val_start_time = time.time()
            """
            It suffices to understand self.valid_one_epoch(val_loader)
            So val_results_computed returns the following:
            
            """
            val_results_computed = self.valid_one_epoch(val_loader)

            val_end_time = time.time()
            val_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(val_end_time - val_start_time))

            self.loss_history.append(val_results_computed["average_loss"])
            self.monitored_metrics_history.append(
                val_results_computed[self.config.monitored_result]
            )

            for result in self.selected_results_val:

                if not isinstance(result, results.SavableResult):
                    continue

                # gets name
                savable_name = result.get_save_name(val_results_computed[result.__class__.__name__])

                if savable_name is None:
                    continue

                self.saved_val_results[savable_name] = val_results_computed[
                    result.__class__.__name__
                ]

            val_reported_results = [
                result.report(val_results_computed[result.__class__.__name__])
                for result in self.selected_results_val
                if isinstance(result, results.ReportableResult)
            ]

            val_result_str = " | ".join(
                [
                    "Validation Epoch: {}".format(self.epoch + 1),
                    *val_reported_results,
                    "Time Elapsed: {}".format(val_elapsed_time),
                ]
            )

            self.log("[VAL RESULT]: {}".format(val_result_str))

            if self.early_stopping is not None:
                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=val_results_computed[self.config.monitored_result]
                )
                """
                Be careful of self.best_loss here, when our monitered_metrics is val_roc_auc, then we should instead write
                self.best_auc = best_score. After which, if early_stop flag becomes True, then we break out of the training loop.
                """

                self.best_val_results[self.config.monitored_result] = best_score
                self.save(
                    "{}_best_{}_fold_{}.pt".format(
                        self.config.model_name, self.config.monitored_result, fold
                    )
                )
                if early_stop:
                    break

            """
            Compute the new best value for all selected ComparableMetric validation results.
            If we find a new best value for the selected monitored result, save the model.
            """

            for result in self.selected_results_val:
                if not isinstance(result, results.ComparableResult):
                    continue

                old_value = self.best_val_results.get(result.__class__.__name__, None)

                if old_value is None:
                    self.best_val_results[result.__class__.__name__] = val_results_computed[
                        result.__class__.__name__
                    ]

                    if result.__class__.__name__ == self.config.monitored_result:
                        self.save(
                            os.path.join(
                                self.save_path,
                                "{}_{}_best_{}_fold_{}.pt".format(
                                    self.date,
                                    self.config.model_name,
                                    self.config.monitored_result,
                                    fold,
                                ),
                            )
                        )

                    continue

                new_value = val_results_computed[result.__class__.__name__]

                if result.compare(old_value, new_value):
                    self.best_val_results[result.__class__.__name__] = new_value

                    if result.__class__.__name__ == self.config.monitored_result:
                        self.log(
                            "Saving epoch {} of fold {} as best weights".format(
                                self.epoch + 1, fold
                            )
                        )
                        self.save(
                            os.path.join(
                                self.save_path,
                                "{}_{}_best_{}_fold_{}.pt".format(
                                    self.date,
                                    self.config.model_name,
                                    self.config.monitored_result,
                                    fold,
                                ),
                            )
                        )
            """
            Usually, we should call scheduler.step() after the end of each epoch. In particular, we need to take note that
            ReduceLROnPlateau needs to step(monitered_metrics) because of the mode argument.
            """
            if self.config.val_step_scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results_computed[self.config.monitored_result])
                else:
                    self.scheduler.step()

            """End of training, epoch + 1 so that self.epoch can be updated."""
            self.epoch += 1

        curr_fold_best_checkpoint = self.load(
            os.path.join(
                self.save_path,
                "{}_{}_best_{}_fold_{}.pt".format(
                    self.date,
                    self.config.model_name,
                    self.config.monitored_result,
                    fold,
                ),
            )
        )
        return curr_fold_best_checkpoint

    def train_one_epoch(self, train_loader):
        """Train one epoch of the model."""
        # set to train mode
        self.model.train()
        self.log("We are not freezing batch norm layers")

        # def set_bn_eval(module):
        #     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        #         module.eval()

        # self.model.apply(set_bn_eval)
        # for name, child in self.model.named_children():
        #     if name.find("BatchNorm") != -1:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = True

        return self.training_results.compute_results(train_loader)

    def valid_one_epoch(self, val_loader):
        """Validate one training epoch."""
        # set to eval mode
        self.model.eval()

        return self.validation_results.compute_results(val_loader)

    def save_model(self, path):
        """Save the trained model."""
        self.model.eval()
        torch.save(self.model.state_dict(), path)

    def save(self, path):
        """Save the weight for the best evaluation loss (and monitored metrics) with corresponding OOF predictions.
        OOF predictions for each fold is merely the best score for that fold."""
        self.model.eval()

        best_results = {
            "best_{}".format(best_result): value
            for (best_result, value) in self.best_val_results.items()
        }

        # print(best_results)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.epoch,
                **best_results,
                **self.saved_val_results,
                "loss_history": self.loss_history,
                self.config.monitored_result: self.monitored_metrics_history,
            },
            path,
        )

    def load(self, path):
        """Load a model checkpoint from the given path."""
        checkpoint = torch.load(path)
        return checkpoint

    def log(self, message):
        """Log a message."""
        if self.config.verbose:
            print(message)
        with open(self.config.paths["log_path"], "a+") as logger:
            logger.write(f"{message}\n")
