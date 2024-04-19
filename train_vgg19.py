from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from collections import defaultdict
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import copy
import argparse
import pandas as pd
import random


def seed_everything(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


seed_everything(42)

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

data_dir = "/DATA/mallampati_data"

parser = argparse.ArgumentParser(description="List fish in aquarium.")
parser.add_argument("--model_name", default="vgg_19", type=str)
parser.add_argument("--learning_rate", default="0.0001", type=float)
parser.add_argument("--batch_size", default="64", type=int)
parser.add_argument("--epoch", default="100", type=int)
args = parser.parse_args()

model_name = args.model_name
learning_rate = args.learning_rate
batch_size = args.batch_size

num_classes = 4
batch_size = args.batch_size
num_epochs = args.epoch

feature_extract = True

results = defaultdict(list)


def train_model(model,
                dataloaders,
                criterion,
                optimizer,
                num_epochs=50,
                is_inception=False):
    since = time.time()
    print("model is : ", model)

    val_acc_history = []
    val_f1_history = []
    val_loss_history = []
    train_acc_history = []
    train_f1_history = []
    train_loss_history = []
    true_labels = []
    pred_labels = []
    epoch_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for fold in range(k_fold_splits):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                true_labels = []
                pred_labels = []

                for inputs, labels in dataloaders[f'fold_{fold+1}'][phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                       
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    true_labels.append(labels.data.cpu().numpy())
                    pred_labels.append(preds.cpu().numpy())

                true_labelse = np.hstack(true_labels)
                pred_labelse = np.hstack(pred_labels)
                f1 = f1_score(true_labelse, pred_labelse, average='macro')

                f1 = f1_score(labels.data, preds)
                epoch_f1 = epoch_f1 / len(dataloaders[phase].dataset)
                epoch_loss = running_loss / len(
                    dataloaders[f'fold_{fold+1}'][phase].dataset)
                epoch_acc = running_corrects.double() / len(
                    dataloaders[f'fold_{fold+1}'][phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, f1))

                if phase == 'val' and f1 > best_f1:
                    best_f1 = f1
                    report = classification_report(true_labelse,
                                                   pred_labelse,
                                                   labels=[0, 1, 2, 3],
                                                   zero_division=0)
                    best_epoch = epoch
                    best_acc = epoch_acc
                    report = classification_report(labels.cpu().data, preds.cpu(), labels=[0,1,2,3],zero_division=0)
                    best_epoch = epoch
                    cf = confusion_matrix(true_labelse, pred_labelse)
                    best_model_wts = copy.deepcopy(model.state_dict())

                    model_save_path = "models_output_base"

                    if not (os.path.exists(model_save_path)):
                        os.makedirs(model_save_path)
                    torch.save(
                        model,
                        f'{model_save_path}/Model_{model_name}______LR_{learning_rate}______Batch_{batch_size}_____Epoch_{num_epochs}.pt'
                    )

                if phase == 'val':
                    val_acc_history.append(epoch_acc.cpu().detach().numpy())
                    val_f1_history.append(f1)
                    val_loss_history.append(epoch_loss)
                if phase == 'train':
                    train_acc_history.append(epoch_acc.cpu().detach().numpy())
                    train_f1_history.append(f1)
                    train_loss_history.append(epoch_loss)

                results_save_path = f"model_result_base/results_____Model_{model_name}______LR_{learning_rate}______Batch_{batch_size}_____Epoch_{num_epochs}/{model_name}_cropped"

                if not (os.path.exists(results_save_path)):
                    os.makedirs(results_save_path)

                np.savetxt(f"{results_save_path}/train_acc_history.csv",
                           train_acc_history,
                           delimiter=",",
                           fmt="%f")
                np.savetxt(f"{results_save_path}/train_loss_history.csv",
                           train_loss_history,
                           delimiter=",",
                           fmt="%f")
                np.savetxt(f"{results_save_path}/train_f1_history.csv",
                           train_f1_history,
                           delimiter=",",
                           fmt="%f")
                np.savetxt(f"{results_save_path}/val_acc_history.csv",
                           val_acc_history,
                           delimiter=",",
                           fmt="%f")
                np.savetxt(f"{results_save_path}/val_loss_history.csv",
                           val_loss_history,
                           delimiter=",",
                           fmt="%f")
                np.savetxt(f"{results_save_path}/val_f1_history.csv",
                           val_f1_history,
                           delimiter=",",
                           fmt="%f")

            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best F1 score: {:.4f}'.format(best_f1))
    print('Best Epoch: ' + str(best_epoch))
    print(report)
    print(cf)

    report_path = f"model_result_base/results_____Model_{model_name}______LR_{learning_rate}______Batch_{batch_size}_____Epoch_{num_epochs}/classification_report.txt"
    with open(report_path, "w") as report_file:
        report_file.write(report)
    report_file.close()

    cf_path = f"model_result_base/results_____Model_{model_name}______LR_{learning_rate}______Batch_{batch_size}_____Epoch_{num_epochs}/cf_report.csv"
    confusion_df = pd.DataFrame(cf)

    confusion_df.to_csv(cf_path)

    model.load_state_dict(best_model_wts)

    return model, val_f1_history, train_f1_history, val_loss_history, train_loss_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.features[0:5].parameters():
            param.requires_grad = False
        for param in model.features[5:].parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name,
                     num_classes,
                     feature_extract,
                     use_pretrained=True):

    model_ft = None
    input_size = 0

    if model_name == "vgg_19":
        """
        VGG11_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


model_ft, input_size = initialize_model(model_name,
                                        num_classes,
                                        feature_extract,
                                        use_pretrained=True)

data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
k_fold_splits = 5
labels = [label for _, label in dataset]

kf = StratifiedKFold(n_splits=k_fold_splits, shuffle=True, random_state=42)
data_dict = {}

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset, labels)):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    batch_size = 64
    train_loader = DataLoader(train_subset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    data_dict[f'fold_{fold_idx + 1}'] = {
        'train': train_loader,
        'val': val_loader
    }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

optimizer_ft = optim.AdamW(params_to_update, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model_ft, vf1, tf1, vloss, tloss = train_model(
    model_ft,
    data_dict,
    criterion,
    optimizer_ft,
    num_epochs,
   )


trainable_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model_ft.parameters() if not p.requires_grad)

print(f"Trainable params: {trainable_params}")
print(f"Non-trainable params: {non_trainable_params}")
