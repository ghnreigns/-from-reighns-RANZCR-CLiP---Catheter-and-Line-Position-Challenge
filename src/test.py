from src.model_blocks import *
import torch.nn as nn
import timm
from src.activations import Swish_Module
from src.config import YAMLConfig
import typing
import gc
from src.utils import *

""" Test two pytorch models are the same """


"""
This is a testing for RANZCR. Note that the model weights will never match unless you go and seed the initializer weights in PyTorch's
kaimin_init. Therefore, if you set pretrained=True, you will realise that the weights will not match at the head level layers because
that is where wour transfer learning stopped. 
"""


class TAWARAMultiHeadResNet200D(nn.Module):
    def __init__(self, out_dims_head: typing.List[int] = [3, 4, 3, 1], pretrained=True):
        """"""
        self.base_name = "resnet200d"
        self.n_heads = len(out_dims_head)
        super(TAWARAMultiHeadResNet200D, self).__init__()

        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=False
        )
        in_features = base_model.num_features

        if pretrained:
            pretrained_model_path = "./input/resnet200d_320_chestx.pth"
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location="cpu")["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)

        # # remove global pooling and head classifier
        base_model.reset_classifier(0, "")

        # # Shared CNN Bacbone
        self.backbone = base_model

        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                Swish_Module(),
                nn.Dropout(0.3),
                nn.Linear(in_features, out_dim),
            )
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


class MultiHeadResNet200D(nn.Module):
    # heads 3431 means total 11 classes, 4 heads, first head corresponds to first 3 classes etc
    def __init__(
        self,
        config,
        out_dims_head: typing.List[int] = [3, 4, 3, 1],
        pretrained=True,
        custom_weights=False,
    ):
        """"""
        self.base_name = "resnet200d"
        self.n_heads = len(out_dims_head)
        super(MultiHeadResNet200D, self).__init__()

        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=pretrained
        )
        in_features = base_model.num_features

        if custom_weights is True:
            print("Loading CUSTOM PRETRAINED WEIGHTS, IF YOU DID NOT CHOOSE THIS, PLEASE RESTART!")
            custom_pretrained_weight_path = config.paths["custom_pretrained_weight"]
            # self.model.load_state_dict(
            #     torch.load(custom_pretrained_weight_path)
            # )
            ### Only for xray pretrained weights ###
            state_dict = dict()
            for k, v in torch.load(custom_pretrained_weight_path, map_location="cpu")[
                "model"
            ].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)

        # # remove global pooling and head classifier
        base_model.reset_classifier(0, "")

        # # Shared CNN Bacbone
        self.backbone = base_model

        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                Swish_Module(),
                # nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(in_features, out_dim),
            )
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y


"""A module for constructing machine learning models."""
import functools
import torch
import geffnet
import timm
from typing import *
from utils import rsetattr
from activations import Swish_Module
from model_blocks import *


class CustomModel(torch.nn.Module):
    """A custom model."""

    def __init__(
        self,
        config: type,
        pretrained: bool = True,
        load_weight: bool = False,
        load_url: bool = False,
        out_dim_heads=[],
        *args,
    ):
        """Construct a custom model."""
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        self.load_weight = load_weight
        self.load_url = load_url
        # TODO: To use args and kwargs for out_dim_heads as it is throwing errors.
        self.args = args
        self.out_dim_heads = out_dim_heads
        self.out_features = None
        self.activation = Swish_Module()

        self.architecture = {
            "backbone": None,
            "bottleneck": None,
            "head": None,
        }

        # def __setattr__(self, name, value):
        #     self.model.__setattr__(self, name, value)

        _model_factory = (
            geffnet.create_model if config.model_factory == "geffnet" else timm.create_model
        )

        self.model = _model_factory(
            # model_weight_path_folder=config.paths["model_weight_path_folder"],
            model_name=config.model_name,
            pretrained=self.pretrained,
            num_classes=11,
        )

        # timm.create_model(
        #     "resnet200d", num_classes=sum(self.out_dim_heads), pretrained=True
        # )

        # load pretrained weight that are not available on timm or geffnet; for example, when NFNet just came out, we do not have timm's pretrained weight
        if self.load_weight:

            # assert (
            #     self.pretrained == False
            # ), "if you are loading custom weight, then pretrained must be set to False"
            print("Loading CUSTOM PRETRAINED WEIGHTS, IF YOU DID NOT CHOOSE THIS, PLEASE RESTART!")
            custom_pretrained_weight_path = config.paths["custom_pretrained_weight"]
            print("Loading custom weights with custom number of classes.")
            # self.model = _model_factory(
            #     model_name=config.model_name,
            #     pretrained=False,
            #     num_classes=self.config.num_classes,
            # )
            # self.model.load_state_dict(
            #     torch.load(custom_pretrained_weight_path)
            # )
            ### Only for xray pretrained weights ###
            state_dict = dict()
            for k, v in torch.load(custom_pretrained_weight_path, map_location="cpu")[
                "model"
            ].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            self.model.load_state_dict(state_dict)
            # self.model.load_state_dict(torch.load(custom_pretrained_weight_path))

        if self.load_url:
            # using torch hub to load url, can be beautified. https://pytorch.org/docs/stable/hub.html
            checkpoint = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-dnf-weights/dm_nfnet_f1-fc540f82.pth"
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location="cpu")
            )

        self.use_custom_layers = True
        NumHeads = len(self.out_dim_heads)  # redundant line

        if self.use_custom_layers is False:
            print("NOT USING CUSTOM LAYERS! BEWARE OF OVERFITTING...")
            last_layer_attr_name, self.out_features, _ = self.get_last_layer()
            last_layer_attr_name = ".".join(last_layer_attr_name)
            rsetattr(
                self.model,
                last_layer_attr_name,
                torch.torch.nn.Linear(self.out_features, config.num_classes),
            )
        else:
            # this is our backbone base, IT IS A SHARED BACKBONE, DO NOT TOUCH!
            # self.architecture['backbone'] = self.model
            # get our number of nodes before the last layer
            # in_features = self.architecture['backbone'].num_features

            if NumHeads == 1:
                print("Single Head Model")
                # timm has reset classifier and get classifier
                self.model.reset_classifier(num_classes=0, global_pool="avg")
                # this is our backbone base, IT IS A SHARED BACKBONE, DO NOT TOUCH!
                self.architecture["backbone"] = self.model

                in_features = self.architecture["backbone"].num_features

                self.single_head_fc = torch.torch.nn.Sequential(
                    torch.torch.nn.Linear(in_features, in_features),
                    self.activation,
                    torch.torch.nn.Dropout(p=0.3),
                    torch.torch.nn.Linear(in_features, config.num_classes),
                )
                self.architecture["head"] = self.single_head_fc

            else:

                self.num_heads = len(self.out_dim_heads)
                print("We are using Multi Head Model with {} Heads".format(self.num_heads))
                in_features = self.model.num_features

                # remove global pooling and head classifier
                self.model.reset_classifier(num_classes=0, global_pool="")
                # Shared CNN Bacbone
                self.architecture["backbone"] = self.model
                # Multi Head
                for i, out_dim in enumerate(self.out_dim_heads):
                    layer_name = f"head_{i}"
                    layer = torch.nn.Sequential(
                        SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                        torch.nn.AdaptiveAvgPool2d(output_size=1),
                        torch.nn.Flatten(start_dim=1),
                        torch.nn.Linear(in_features, in_features),
                        self.activation,
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(in_features, out_dim),
                    )
                    setattr(self, layer_name, layer)

    def forward(self, input_neurons):
        """Define the computation performed at every call."""
        if self.use_custom_layers is False:
            output_predictions = self.model(input_neurons)
        else:
            if len(self.out_dim_heads) > 1:
                print("s")
                output_logits_backbone = self.architecture["backbone"](input_neurons)
                multi_outputs = [
                    getattr(self, f"head_{i}")(output_logits_backbone)
                    for i in range(self.num_heads)
                ]
                output_predictions = torch.cat(multi_outputs, axis=1)
            else:  # only single head
                output_logits_backbone = self.architecture["backbone"](input_neurons)
                output_predictions = self.architecture["head"](output_logits_backbone)

        return output_predictions

    def get_last_layer(self):
        last_layer_name = None
        for name, param in self.model.named_modules():
            last_layer_name = name

        last_layer_attributes = last_layer_name.split(".")  # + ['in_features']
        linear_layer = functools.reduce(getattr, last_layer_attributes, self.model)
        # reduce applies to a list recursively and reduce
        in_features = functools.reduce(getattr, last_layer_attributes, self.model).in_features
        return last_layer_attributes, in_features, linear_layer


## forward test

if __name__ == "__main__":

    yaml_config = YAMLConfig("./config_debug.yaml")
    seed_all(seed=yaml_config.seed)
    import timm

    HN_MODEL = CustomModel(
        config=yaml_config,
        pretrained=True,
        load_weight=True,
        load_url=False,
        out_dim_heads=[3, 4, 3, 1],
    )
    HN_MODEL_DUPLICATE = CustomModel(
        config=yaml_config,
        pretrained=True,
        load_weight=True,
        load_url=False,
        out_dim_heads=[3, 4, 3, 1],
    )
    # HN_MODEL = HN_MODEL.eval()

    """ 
    Initiate TAWARA's model TAWARAMultiHeadResNet200D, not at the same time. We test TAMARA_MODEL first and get a forward value of 0.0476.
    Then we test TAWARA_MODEL_DUPLICATE which is the exact same model and test to get 0.0476 OR 0.0348

    Then we test TAWARA_MODEL_NEW which is a slight variation in the construction but same architecture. Setting eval is optional, but must be consistent.
    CAUTION: LEAVE THE HN_MODEL intact as it needs to run sequentially, so the GPU behind runs HN_MODEL first, so if you comment it out
    the values might change.
    """

    # TAWARA_MODEL = TAWARAMultiHeadResNet200D(
    #     out_dims_head=[3, 4, 3, 1], pretrained=True
    # )

    # TAWARA_MODEL_DUPLICATE = TAWARAMultiHeadResNet200D(
    #     out_dims_head=[3, 4, 3, 1], pretrained=True
    # )
    # TAWARA_MODEL = TAWARA_MODEL.eval()

    # TAWARA_MODEL_NEW = MultiHeadResNet200D(
    #     yaml_config, [3, 4, 3, 1], pretrained=True, custom_weights=True
    # )
    # TAWARA_MODEL_NEW = TAWARA_MODEL_NEW.eval()

    ### Find layers like batchnorm or conv2d ###
    # print(find_layer(HN_MODEL))

    ### Get weight of each layer ###
    # print(get_weight(TAWARA_MODEL, 1))

    ### Compare if two pretrained model are equal | if heads are changed drastically, then it will be different ###
    # print(compare_models(TAWARA_MODEL_DUPLICATE, TAWARA_MODEL))

    def forward_test(x, model):
        y1 = model(x)
        print("[forward test]")
        print("input:\t{}\noutput:\t{}".format(x.shape, y1.shape))
        print("output value", y1[0][0])

    x = torch.rand(1, 3, 256, 256)
    print(forward_test(x, model=HN_MODEL_DUPLICATE))

    # RESNET_FREEZE_BN = timm.create_model("resnet50", pretrained=True)
    # RESNET_UNFREEZE_BN = timm.create_model("resnet50", pretrained=True)
    # # RESNET_FREEZE_BN.apply(set_bn_eval)
    # # print(find_layer(RESNET_FREEZE_BN))

    # x = torch.rand(1, 3, 256, 256)
    # with torch.no_grad():
    #     y1 = RESNET_FREEZE_BN(x)
    #     y2 = RESNET_UNFREEZE_BN(x)
    # print("[forward test]")
    # print("input:\t{}\noutput:\t{}".format(x.shape, y1.shape))
    # print("output value", y1[0][0])

    # print("[forward test]")
    # print("input:\t{}\noutput:\t{}".format(x.shape, y2.shape))
    # print("output value", y2[0][0])

    # del RESNET_UNFREEZE_BN, RESNET_FREEZE_BN
    # del x
    # del y1, y2
    # gc.collect()
