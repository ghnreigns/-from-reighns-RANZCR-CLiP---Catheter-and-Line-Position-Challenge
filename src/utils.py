"""Some utility functions."""
import glob
import os
import random
import functools
from collections import Counter
from typing import List, Optional, Tuple
from torchsummary import summary

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


#
# for module in model.modules():
#     # print(module)
#     if isinstance(module, nn.BatchNorm2d):
#         if hasattr(module, "weight"):
#             module.weight.requires_grad_(False)
#         if hasattr(module, "bias"):
#             module.bias.requires_grad_(False)
#         module.eval()
# Find layers
def find_layer(_model):
    for module in _model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            print(module)
            # if hasattr(module, "weight"):
            #     module.weight.requires_grad_(False)
            # if hasattr(module, "bias"):
            #     module.bias.requires_grad_(False)
            # module.eval()


# get weights of each layer
def get_weight(_model, num_layers=None):  # number of layers you want to test
    counter = 0
    for key_item in _model.state_dict().items():
        layer_name, layer_weight = key_item
        print()
        print(layer_name)
        print()
        print(layer_weight[0])
        counter += 1
        if num_layers is not None and counter > num_layers:
            break
        if ".layer1.0.bn1.weight" in layer_name:
            print(layer_name, layer_weight[0])


# compare two models
def compare_models(model_1, model_2):
    models_differ = 0
    print(
        "Reminder that we need to set pretrained to True if not the weights will be randomly initialized."
    )
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                # print(key_item_1)
                print("Mismatch found at", key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")


# see pytorch model summary using torchsummary
def torchsummary_wrapper(model, image_size: Tuple):

    model_summary = summary(model, image_size)
    return model_summary


## set and get attribute dynamically
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def check_file_type(image_folder_path, allowed_extensions: Optional[List] = None):
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".png", ".jpeg"]

    no_files_in_folder = len(glob.glob(os.path.join(image_folder_path, "*")))
    extension_type = ""
    no_files_allowed = 0

    for ext in allowed_extensions:
        no_files_allowed = len(
            glob.glob(os.path.join(image_folder_path, "*.{}".format(ext)))
        )
        if no_files_allowed > 0:
            extension_type = ext
            break

    assert (
        no_files_in_folder == no_files_allowed
    ), "The extension in the folder should all be the same, but found more than one extensions"
    return extension_type


def get_file_type(image_folder_path: str, allowed_extensions: Optional[List] = None):
    """Get the file type of images in a folder."""
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".png", ".jpeg"]

    file_list = os.listdir(image_folder_path)
    extension_type = [os.path.splitext(file)[-1].lower() for file in file_list]
    extension_dict = Counter(extension_type)
    assert (
        len(extension_dict.keys()) == 1
    ), "The extension in the folder should all be the same, "
    "but found {} extensions".format(extension_dict.keys)
    extension_type = list(extension_dict.keys())[0]
    assert extension_type in allowed_extensions
    return extension_type


""" Consider modifying this function below to check if the dataframe's
image id column has extension or not """


def check_df_ext(
    df: pd.DataFrame, col_name: str, allowed_extensions: Optional[List] = None
):
    """Get the image file extension used in a data frame."""
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".png", ".jpeg"]
    # check if the col has an extension, this is tricky.
    # if no extension, it gives default ""
    image_id_list = df[col_name].tolist()
    extension_type = [
        # Review Comments: os.path.splitext is guaranteed to return a 2-tuple,
        # so no need to use -1 index.
        os.path.splitext(image_id)[1].lower()
        for image_id in image_id_list
    ]

    assert (
        len(set(extension_type)) == 1
    ), "The extension in the image id should all be the same"
    if "" in extension_type:
        return False
    assert list(set(extension_type))[0] in allowed_extensions
    return True


# Check the image folder for corrupted images.


def image_corruption(image_folder_path, img_type):
    """Find images in a folder that are corrupted."""
    corrupted_images = filter(
        lambda path_name: cv2.imread(path_name) is None,
        glob.glob(os.path.join(image_folder_path, img_type)),
    )
    for image_name in corrupted_images:
        print("This image {} is corrupted!".format(os.path.basename(image_name)))


def check_image_size(image_folder_path, height=None, width=None):
    """Count the number of images having differing dimensions."""
    total_img_list = glob.glob(os.path.join(image_folder_path, "*"))
    counter = 0
    for image in tqdm(total_img_list, desc="Checking in progress"):
        try:
            img = cv2.imread(image)

            # Review Comments:
            #
            # I assume you were trying to initialize width and height
            # if they are not defined by the caller. I have rewritten
            # your code to do this successfully - before you were just
            # comparing the height and width of each image with
            # itself.
            if height is None:
                height = img.shape[1]

            if width is None:
                width = img.shape[0]

            if not (height == img.shape[1] and width == img.shape[0]):
                counter += 1
        # Review Comments: What exception are you trying to catch here?
        # In general, you should not have a bare except block.
        except:
            print("this {} is corrupted".format(image))
            continue
    return counter


def seed_all(seed: int = 1930):
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id):
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
