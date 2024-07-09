import os
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as transforms
import pandas as pd

NORMALIZATION_STATISTICS = {"self_learning_cubes_32": [[0.11303308354465243, 0.12595135887180803]],
                            "self_learning_cubes_64": [[0.11317437834743148, 0.12611378817031038]],
                            "lidc": [[0.23151727, 0.2168428080133056]],
                            "luna_fpr": [[0.18109835972793722, 0.1853707675313153]],
                            "lits_seg": [[0.46046468844492944, 0.17490586272419967]],
                            "pe": [[0.26125720740546626, 0.20363551346695796]]}


# -------------------------------------Data augmentation-------------------------------------
class Augmentation():
    def __init__(self, normalize):
        if normalize.lower() == "imagenet":
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif normalize.lower() == "chestx-ray":
            self.normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        elif normalize.lower() == "none":
            self.normalize = None
        else:
            print("mean and std for [{}] dataset do not exist!".format(normalize))
            exit(-1)

    def get_augmentation(self, augment_name, mode):
        try:
            aug = getattr(Augmentation, augment_name)
            return aug(self, mode)
        except:
            print("Augmentation [{}] does not exist!".format(augment_name))
            exit(-1)

    def basic(self, mode):
        transformList = []
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def _basic_crop(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
        else:
            transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_224(self, mode):
        transCrop = 224
        return self._basic_crop(transCrop, mode)

    def _basic_resize(self, size, mode="train"):
        transformList = []
        transformList.append(transforms.Resize(size))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_resize_224(self, mode):
        size = 224
        return self._basic_resize(size, mode)

    def _basic_crop_rot(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
            transformList.append(transforms.RandomRotation(7))
        else:
            transformList.append(transforms.CenterCrop(transCrop))

        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_rot_224(self, mode):
        transCrop = 224
        return self._basic_crop_rot(transCrop, mode)

    def _full(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full(transCrop, transResize, mode)

    def full_448(self, mode):
        transCrop = 448
        transResize = 512
        return self._full(transCrop, transResize, mode)

    def _full_colorjitter(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_colorjitter_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full_colorjitter(transCrop, transResize, mode)


from torch.utils.data import Dataset


# --------------------------------------------Downstream ChestX-ray14-------------------------------------------


class Covidx(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform):
        self.data_dir = data_dir
        self.phase = phase

        self.classes = ['normal', 'positive', 'pneumonia', 'COVID-19']
        self.class2label = {c: i for i, c in enumerate(self.classes)}

        # collect training/testing files
        if phase == 'train':
            with open(os.path.join(data_dir, 'train_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.datalist = list()
        for line in lines:
            patient_id, fname, label, source = line.split(' ')
            if phase in ('train', 'val'):
                self.datalist.append((os.path.join(data_dir, 'train', fname), label))
            else:
                self.datalist.append((os.path.join(data_dir, 'test', fname), label))

        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        fpath, label = self.datalist[index]
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        label = self.class2label[label]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class CheXpert(Dataset):
    '''
    Reference:
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''

    def __init__(self,
                 csv_path,
                 image_root_path='',
                 class_index=0,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transform=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 mode='train',
                 heatmap_path=None,
                 pretraining=False,
                 ):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        # self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        # self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')

        else:
            self.heatmap = None

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

            # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:  # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index

        self.transform = transform

        self._images_list = [image_root_path + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                print('-' * 30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                print('-' * 30)
            else:
                print('-' * 30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1] / (
                            self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                    print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print('-' * 30)
        self.pretraining = pretraining

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        if self.heatmap is None:
            image = Image.open(self._images_list[idx]).convert('RGB').resize((224, 224))

            image = self.transform(image)

            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return image, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            image = Image.open(self._images_list[idx]).convert('RGB').resize((224, 224))

            image, heatmap = self.transform(image, heatmap)
            heatmap = heatmap.permute(1, 2, 0)
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return [image, heatmap], label




class CheXpertDatasetFCRO(Dataset):
    def __init__(
        self,
        csv_path,
        image_root_path,
        target_labels=None,
        sensitive_attribute="Sex",
        shuffle=True,
        transform=None,
    ):

        if target_labels is None:
            raise RuntimeError("Please give your target label.")
            
        self.image_root_path = image_root_path
        self.target_labels = target_labels if isinstance(target_labels, list) else [target_labels]
        self.sensitive_attributes = sensitive_attribute
        self.transform = transform

        if isinstance(self.sensitive_attributes, str):
            self.sensitive_attributes = [self.sensitive_attributes]

        self.df = pd.read_csv(csv_path)

        # impute missing values
        for col in self.target_labels:
            if col in ["Edema", "Atelectasis"]:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ["Cardiomegaly", "Consolidation", "Pleural Effusion"]:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in [
                "No Finding",
                "Enlarged Cardiomediastinum",
                "Lung Opacity",
                "Lung Lesion",
                "Pneumonia",
                "Pneumothorax",
                "Pleural Other",
                "Fracture",
                "Support Devices",
            ]:  # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self.num_imgs = len(self.df)
        
        if self.sensitive_attributes is not None:
            for sa in self.sensitive_attributes:
                if sa == "Race":
                    self.df[sa] = self.df[sa].apply(lambda x: 1 if "White" in x else 0)
                elif sa == "Sex":
                    self.df[sa] = self.df[sa].apply(lambda x: 1 if x == "Male" else 0)
                elif sa == "Age":
                    self.df[sa] = self.df[sa].apply(lambda x: 1 if x > 60 else 0)

        # shuffle data
        if shuffle:
            data_index = list(range(self.num_imgs))
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        self.images_list = [
            os.path.join(self.image_root_path, path) for path in self.df["Path"].tolist()
        ]
        self.targets = self.df[self.target_labels].values.tolist()
        if self.sensitive_attributes is not None:
            self.a_dict = {}
            for attribute in self.sensitive_attributes:
                self.a_dict[attribute] = self.df[attribute].values.tolist()

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx]).convert("RGB")


        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(self.targets[idx]).view(-1)
        
        if self.sensitive_attributes == None:
            return image, target
        
        a = {}
        for k, v in self.a_dict.items():
            a[k] = torch.tensor(v[idx]).view(-1)

        return image, target, a[self.sensitive_attributes[0]]


class CheXpertDatasetSubgroup(Dataset):
    def __init__(
        self,
        csv_path,
        image_root_path,
        target_labels=None,
        sensitive_attribute="Race",
        shuffle=True,
        transform=None,
        subgroup=None
    ):

        if target_labels is None:
            target_labels = [
                "No Finding",
                "Pleural Effusion",
            ]

        self.image_root_path = image_root_path
        self.target_labels = target_labels if isinstance(target_labels, list) else [target_labels]
        self.sensitive_attributes = sensitive_attribute
        self.transform = transform

        if isinstance(self.sensitive_attributes, str):
            self.sensitive_attributes = [self.sensitive_attributes]

        self.df = pd.read_csv(csv_path)
        if "Pleural Effusion" in target_labels and len(target_labels) == 2:
            if subgroup == "male_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "male_pleural":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['Pleural Effusion'] == 1.0)]
            elif subgroup == "female_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "female_pleural":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['Pleural Effusion'] == 1.0)]
            elif subgroup is None:
                self.df = self.df
            else:
                raise NotImplementedError("There is no such subgroup of features")
        elif "Pneumonia" in target_labels and len(target_labels) == 2:
            if subgroup == "male_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "male_pneumonia":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['Pneumonia'] == 1.0)]
            elif subgroup == "female_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "female_pneumonia":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['Pneumonia'] == 1.0)]
            elif subgroup is None:
                self.df = self.df
            else:
                raise NotImplementedError("There is no such subgroup of features")
        elif "Lesion" in target_labels and len(target_labels) == 2:
            if subgroup == "male_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "male_lesion":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['Lung Lesion'] == 1.0)]
            elif subgroup == "female_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "female_lesion":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['Lung Lesion'] == 1.0)]
            elif subgroup is None:
                self.df = self.df
            else:
                raise NotImplementedError("There is no such subgroup of features")
        elif "Edema" in target_labels and len(target_labels) == 2:
            if subgroup == "male_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "male_edema":
                self.df = self.df[(self.df['Sex'] == 'Male') & (self.df['Edema'] == 1.0)]
            elif subgroup == "female_nofinding":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['No Finding'] == 1.0)]
            elif subgroup == "female_edema":
                self.df = self.df[(self.df['Sex'] == 'Female') & (self.df['Edema'] == 1.0)]
            elif subgroup is None:
                self.df = self.df
            else:
                raise NotImplementedError("There is no such subgroup of features")
        else:
            raise NotImplementedError("There is no such diease.")

        # impute missing values
        for col in self.target_labels:
            if col in ["Edema", "Atelectasis"]:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ["Cardiomegaly", "Consolidation", "Pleural Effusion"]:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in [
                "No Finding",
                "Enlarged Cardiomediastinum",
                "Lung Opacity",
                "Lung Lesion",
                "Pneumonia",
                "Pneumothorax",
                "Pleural Other",
                "Fracture",
                "Support Devices",
            ]:  # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self.num_imgs = len(self.df)
        
        if self.sensitive_attributes is not None:
            for sa in self.sensitive_attributes:
                if sa == "Race":
                    self.df[sa] = self.df[sa].apply(lambda x: 1 if "White" in x else 0)
                elif sa == "Sex":
                    self.df[sa] = self.df[sa].apply(lambda x: 1 if x == "Male" else 0)
                elif sa == "Age":
                    self.df[sa] = self.df[sa].apply(lambda x: 1 if x > 60 else 0)

        # shuffle data
        if shuffle:
            data_index = list(range(self.num_imgs))
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        self.images_list = [
            os.path.join(self.image_root_path, path) for path in self.df["Path"].tolist()
        ]
        self.targets = self.df[self.target_labels].values.tolist()
        if self.sensitive_attributes is not None:
            self.a_dict = {}
            for attribute in self.sensitive_attributes:
                self.a_dict[attribute] = self.df[attribute].values.tolist()

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(self.targets[idx]).view(-1)
        
        # if self.sensitive_attributes == None:
        #     return image, target
        
        a = {}
        for k, v in self.a_dict.items():
            a[k] = torch.tensor(v[idx]).view(-1)

        return image, target, a[self.sensitive_attributes[0]]