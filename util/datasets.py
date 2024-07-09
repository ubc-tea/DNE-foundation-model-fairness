# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.dataloader_med import CheXpertDatasetFCRO, CheXpertDatasetSubgroup

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_dataset_chest_xray(split, args, subgroup=None):
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ])

    if args.dataset == 'chexpert':
        if split == 'train':
            mode = 'train'
        else:
            mode = 'valid'

        if args.disease == "Pleural":
            target_labels = [
                    "No Finding",
                    "Pleural Effusion",
                ]
        elif args.disease == "Pneumonia":
            target_labels = [
                    "No Finding",
                    "Pneumonia",
                ]
        elif args.disease == "Lesion":
            target_labels = [
                "No Finding",
                "Lung Lesion",
                ]
        elif args.disease == "Edema":
            target_labels = [
                "No Finding",
                "Edema",
                ]
        else:
            raise NotImplementedError()
        
        if subgroup is None:
            dataset = CheXpertDatasetFCRO(
                csv_path=os.path.join(args.csv_path, "{}.csv".format(mode)),
                image_root_path=args.data_path,
                target_labels=target_labels,
                sensitive_attribute="Sex",
                shuffle=True,
                transform=transform,
            )
        else:
            dataset = CheXpertDatasetSubgroup(
                csv_path=os.path.join(args.csv_path, "{}.csv".format(mode)),
                image_root_path=args.data_path,
                target_labels=target_labels,
                sensitive_attribute="Sex",
                shuffle=True,
                transform=transform,
                subgroup=subgroup
            )
    else:
        raise NotImplementedError()
    print(dataset)

    return dataset




def build_transform(is_train, args):

    if args.norm_stats is not None:
        if args.norm_stats == 'imagenet':
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            raise NotImplementedError
    else:
        try:
            if args.dataset == 'chestxray' or args.dataset == 'covidx' or args.dataset == 'chexpert':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)
            elif args.dataset == 'imagenet':
                mean = IMAGENET_DEFAULT_MEAN
                std = IMAGENET_DEFAULT_STD
            elif args.dataset == 'retina':
                mean = (0.5056, 0.5056, 0.5056)
                std = (0.252, 0.252, 0.252)
        except:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD


    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

