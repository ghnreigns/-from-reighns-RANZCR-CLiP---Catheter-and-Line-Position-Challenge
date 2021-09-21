from dataset import CustomDataset, CustomDataLoader
from model import CustomModel
import transforms
import glob
from src.utils import *
import timm
from src.config import YAMLConfig
import albumentations
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2

# TODO: ToTensorV2() should always be added when using albumentations with PyTorch even though one may not need it everytime if you transposed your data properly in Dataset.

# SIN's version batch_size = 1 # if we using sin because of the code view
def tta_inference_func(config, model, test_loader):
    model.to(config.device)
    model.eval()
    bar = tqdm(test_loader)
    PREDS = []
    LOGITS = []

    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(config.device)
            x = torch.stack([x, x.flip(-1)], 0)  # hflip
            x = x.view(-1, 3, config.device.image_size, config.device.image_size)
            logits = model(x)
            logits = logits.view(config.val_batch_size, 2, -1).mean(1)
            PREDS += [logits.sigmoid().detach().cpu()]
            LOGITS.append(logits.cpu())
        PREDS = torch.cat(PREDS).cpu().numpy()

    return PREDS


def inference_by_fold(config, model, state_dicts, test_loader):
    model.to(config.device)
    model.eval()
    probs = []

    with torch.no_grad():
        all_folds_preds = []
        for fold_num, state in enumerate(state_dicts):
            if "model_state_dict" not in state:
                model.load_state_dict(state)
            else:
                model.load_state_dict(state["model_state_dict"])

            current_fold_preds = []
            for data in tqdm(test_loader, position=0, leave=True):
                img_ids, images, labels = data
                images = images.to(config.device)
                logits = model(images)

                sigmoid_preds = logits.sigmoid().detach().cpu().numpy()
                current_fold_preds.append(sigmoid_preds)

            current_fold_preds = np.concatenate(current_fold_preds, axis=0)
            all_folds_preds.append(current_fold_preds)
        avg_preds = np.mean(all_folds_preds, axis=0)
    return avg_preds


def inference_tta_by_fold(config, model, states_dicts, test_loader, tta_test_loader, **kwargs):
    MORE_TTA_DICT = kwargs  # TODO: If you want more than one TTA, add a dictionary and unpack.
    model.to(config.device)
    model.eval()
    probs = []

    with torch.no_grad():
        all_folds_preds = []
        for fold_num, state in enumerate(states_dicts):
            if "model_state_dict" not in state:
                model.load_state_dict(state)
            else:
                model.load_state_dict(state["model_state_dict"])

            current_fold_preds = []
            # https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
            for index, (data, tta_data) in enumerate(zip(test_loader, tta_test_loader)):
                img_ids, images, labels = data
                img_ids_tta, images_tta, labels_tta = tta_data
                images = images.to(config.device)
                images_tta = images_tta.to(config.device)

                logits = model(images)
                logits_tta = model(images_tta)
                sigmoid_preds = logits.sigmoid().detach().cpu().numpy()
                tta_sigmoid_preds = logits_tta.sigmoid().detach().cpu().numpy()

                final_preds = np.mean([sigmoid_preds, tta_sigmoid_preds], axis=0)
                current_fold_preds.append(final_preds)

            current_fold_preds = np.concatenate(current_fold_preds, axis=0)
            all_folds_preds.append(current_fold_preds)

        avg_preds = np.mean(all_folds_preds, axis=0)

    return avg_preds


def LoadTestSet(test_df: pd.DataFrame, config):
    """Train the model on the given fold."""
    model = CustomModel(
        config=config,
        pretrained=False,
        load_weight=False,
        load_url=False,
        out_dim_heads=[3, 4, 3, 1],
    )
    model.to(config.device)
    model_summary = torchsummary_wrapper(model, (3, config.image_size, config.image_size))
    print("model summary: {}".format(model_summary))

    def RANZCR_TEST_AUG(image_size=640):
        transforms_dict = {
            "transforms_test": albumentations.Compose(
                [
                    albumentations.Resize(image_size, image_size),
                    albumentations.Normalize(mean=[0.4887381077884414], std=[0.23064819430546407]),
                    ToTensorV2(),
                ]
            ),
            "tta_hflip": albumentations.Compose(
                [
                    albumentations.HorizontalFlip(p=1.0),
                    albumentations.Resize(image_size, image_size),
                    albumentations.Normalize(mean=[0.4887381077884414], std=[0.23064819430546407]),
                    ToTensorV2(),
                ]
            ),
            # "tta_vflip": albumentations.Compose(
            #     [
            #         albumentations.VerticalFlip(p=1.0),
            #         albumentations.Resize(image_size, image_size),
            #         ToTensorV2(),
            #     ]
            # ),
            # "tta_motion_blur": albumentations.Compose(
            #     [
            #         albumentations.MotionBlur(p=1.0),
            #         albumentations.Resize(image_size, image_size),
            #         ToTensorV2(),
            #     ]
            # ),
            # "tta_shiftscalerotate": albumentations.Compose(
            #     [
            #         albumentations.ShiftScaleRotate(rotate_limit=0, p=1.0),
            #         albumentations.Resize(image_size, image_size),
            #         ToTensorV2(),
            #     ]
            # ),
        }
        return transforms_dict

    all_preds = {}
    transform_dict = RANZCR_TEST_AUG(image_size=config.image_size)

    weights = [
        r"C:\Users\reighns\ranzcr\stored_models\Uploaded_2021-09-15_resnet200d_best_multi_class_roc_auc_score_fold_2.pt"
    ]
    state_dicts = [
        torch.load(checkpoint_path, map_location=torch.device("cuda"))
        for checkpoint_path in glob.glob("./stored_models/*.pt")
    ]

    for aug, aug_param in transform_dict.items():
        test_dataset = CustomDataset(config, df=test_df, transforms=aug_param, mode="test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        predictions = inference_by_fold(
            config=config, model=model, state_dicts=state_dicts, test_loader=test_loader
        )

        all_preds[aug] = predictions

        # Repeatedly overwrite test_df

        test_df[config.class_col_name] = predictions
        test_df.to_csv(f"submission_{aug}.csv", index=False)
        # test_df[["id", "target"]].to_csv(f"submission_{aug}.csv", index=False)
        test_df.head()

        # plt.figure(figsize=(12, 6))
        # plt.hist(test_df.target, bins=100)
    return all_preds


if __name__ == "__main__":

    CURRENT_MODEL = "resnet200d"
    yaml_config = YAMLConfig("./config/config_RANZCR_resnet200d.yaml")
    # MODELS = {
    #     "resnet200d": "/content/reighns/config_RANZCR_resnet200d.yaml",
    # }

    # yaml_config = YAMLConfig(MODELS[CURRENT_MODEL])

    seed_all(seed=yaml_config.seed)
    df_test = pd.read_csv("./data/sample_submission.csv")
    LoadTestSet(df_test, yaml_config)
