from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import models_vit

from util.dataloader_med import CheXpertDatasetFCRO

class DataLoaderFactory:
    @staticmethod
    def create_dataloader(csv_path, image_root_path, target_labels, sensitive_attribute, transform, batch_size, shuffle=True, mode=None, sheet=None, patch_indices=None, num_workers=8):
        dataset = CheXpertDatasetFCRO(
            csv_path=csv_path,
            image_root_path=image_root_path,
            target_labels=target_labels,
            sensitive_attribute=sensitive_attribute,
            shuffle=shuffle,
            transform=transform,
            mode=mode,
            sheet=sheet,
            patch_indices=patch_indices
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader

class ModelFactory:
    @staticmethod
    def create_model(model_name):
        if model_name == "resnet":
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        elif model_name == "vit_base_patch16":
            pretrained_path = "./checkpoints/vit-b_CXR_0.5M_mae_pretrain.pth"
            global_pool = True
            model = models_vit.__dict__[model_name](
                img_size=224,
                num_classes=2,
                drop_rate=0,
                drop_path_rate=0.1,
                global_pool=True,
            )
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % pretrained_path)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            if global_pool:
                for k in ['fc_norm.weight', 'fc_norm.bias']:
                    try:
                        del checkpoint_model[k]
                    except:
                        pass
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
            for param in model.parameters():
                param.requires_grad = False
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                raise NameError("The model does not have a recognizable classifier head")
        elif model_name == "densenet":
            pretrained_path = "./checkpoints/densenet121_CXR_0.3M_mae_pretrain.pth"
            model = models.__dict__["densenet121"](num_classes=2)
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % pretrained_path)
            if 'state_dict' in checkpoint.keys():
                checkpoint_model = checkpoint['state_dict']
            elif 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            if True:
                state_dict = checkpoint_model
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    if 'model.encoder.' in key:
                        new_key = key.replace('model.encoder.', '')
                        new_state_dict[new_key] = value
                checkpoint_model = new_state_dict
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
            for param in model.parameters():
                param.requires_grad = False
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                raise NameError("The model does not have a recognizable classifier head")
        else:
            raise NotImplementedError()
        return model

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, num_epochs=50):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels, genders in tqdm(self.train_loader):
                inputs, labels, genders = inputs.to(self.device), labels.to(self.device), genders.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, genders.view(-1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == genders.view(-1).data)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{self.num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, genders in tqdm(self.test_loader):
                inputs, labels, genders = inputs.to(self.device), labels.to(self.device), genders.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, genders.view(-1))
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == genders.view(-1).data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(genders.cpu().numpy())
        epoch_loss = running_loss / len(self.test_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.test_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        print(f'Epoch {epoch+1}/{self.num_epochs} Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    fix_seed(1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_loader = DataLoaderFactory.create_dataloader(
        csv_path="./sampled_data/train.csv",
        image_root_path="../DataSets",
        target_labels=["No Finding"],
        sensitive_attribute="Sex",
        transform=train_transform,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoaderFactory.create_dataloader(
        csv_path="./sampled_data/valid.csv",
        image_root_path="../DataSets",
        target_labels=["No Finding"],
        sensitive_attribute="Sex",
        transform=test_transform,
        batch_size=32,
        shuffle=False
    )

    model = ModelFactory.create_model(model_name="vit_base_patch16")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device)
    trainer.train()

    torch.save(model.state_dict(), "./sa_cls.pt")
