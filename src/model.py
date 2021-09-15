"""A module for constructing machine learning models."""
import functools
import torch
import geffnet
import timm
from typing import *
from src.utils import rsetattr
from src.activations import Swish_Module
from src.model_blocks import *


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

        def __setattr__(self, name, value):
            self.model.__setattr__(self, name, value)

        _model_factory = (
            geffnet.create_model if config.model_factory == "geffnet" else timm.create_model
        )

        self.model = _model_factory(
            # model_weight_path_folder=config.paths["model_weight_path_folder"],
            model_name=config.model_name,
            pretrained=self.pretrained,
            num_classes=config.num_classes,
        )

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
        # TODO: rename output_predictions to output_logits

        if self.use_custom_layers is False:
            output_predictions = self.model(input_neurons)
        else:
            if len(self.out_dim_heads) > 1:
                output_logits_backbone = self.architecture["backbone"](input_neurons)
                multi_outputs = [
                    getattr(self, f"head_{i}")(output_logits_backbone)
                    for i in range(self.num_heads)
                ]

                ### Debug Testing ###
                # If batch size is 8, then head_1 should have shape of (8, 3) etc
                (
                    head_1_output_logits,
                    head_2_output_logits,
                    head_3_output_logits,
                    head_4_output_logits,
                ) = multi_outputs
                assert head_1_output_logits.shape[1] == 3
                assert head_2_output_logits.shape[1] == 4
                assert head_3_output_logits.shape[1] == 3
                assert head_4_output_logits.shape[1] == 1

                # Concatenate all 4 heads to form shape (BS, 11) where BS = batch_size
                output_predictions = torch.cat(multi_outputs, axis=1)
                assert output_predictions.shape[1] == 11
                # print(output_predictions)

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
