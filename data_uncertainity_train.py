import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.distributions as tdist
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import sys
import os
import time
import argparse
import pandas as pd
import random
from sklearn.utils.class_weight import compute_class_weight


def seed_everything(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


seed_everything(42)

mode = 'Train'
feature_extract = True

parser = argparse.ArgumentParser(description="List of parameters")
parser.add_argument("--model_name", default="alexnet", type=str)
parser.add_argument("--learning_rate", default="0.0001", type=float)
parser.add_argument("--batch_size", default="64", type=int)
parser.add_argument("--epoch", default = "100",type=int)
args = parser.parse_args()

model_name = args.model_name
learning_rate = args.learning_rate
batch_size = args.batch_size
data_dir = "/DATA/mallampati_data"

use_pretrained=True

T_monte_carlo = 20
loss_div = [0.3,0.7]
numclasses = 4

data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val','test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val','test']}

num_epochs = args.epoch
best_acc = 0.0
best_f1 = 0.0
shuffle_dataset = True

dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
labels = [label for _, label in dataset]

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

data_dict = {}

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset, labels)):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    data_dict[f'fold_{fold_idx + 1}'] = {'train': train_loader, 'val': val_loader}

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.features[0:5].parameters():
            param.requires_grad = False
        for param in model.features[5:].parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset, labels)):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    data_dict[f'fold_{fold_idx + 1}'] = {'train': train_loader, 'val': val_loader}

class AleatoricModel(nn.Module):
    def __init__(self, eff_model):
        super(AleatoricModel, self).__init__()
        self.eff_model = eff_model

        if model_name=="efficientnet":
            self.eff_model.classifier[-1]=torch.nn.Identity()
            self.logits_layer = nn.Linear(1280,4,bias=True)
            self.variance_layer = nn.Linear(1280,1,bias=True)
        
        elif model_name=="alexnet":
            self.eff_model.classifier[-1]=torch.nn.Identity()
            self.logits_layer = nn.Linear(4096,4,bias=True)
            self.variance_layer = nn.Linear(4096,1,bias=True)

        elif model_name=="vgg":
            self.eff_model.classifier[-1]=torch.nn.Identity()
            self.logits_layer = nn.Linear(4096,4,bias=True)
            self.variance_layer = nn.Linear(4096,1,bias=True)

        elif model_name=="convnext":
            self.eff_model.classifier[-1]=torch.nn.Identity()
            self.logits_layer = nn.Linear(768,4,bias=True)
            self.variance_layer = nn.Linear(768,1,bias=True)

        self.softplus = nn.Softplus()
        self.variance = None
        self.variance_pre = None

    def forward(self,x):
        eff_out = self.eff_model(x)
        variance_pre = self.variance_layer(eff_out)
        variance = self.softplus(variance_pre)
        self.variance = variance
        self.variance_pre = variance_pre
        logits = self.logits_layer(eff_out)
        logit_variance = torch.cat((logits,variance),axis=1)
        return logit_variance, logits, variance, variance_pre
        
class bayesian_categorical_crossentropy(nn.Module):
    def __init__(self, T, num_classes):
        super(bayesian_categorical_crossentropy, self).__init__()
        self.T = T
        self.ELU = nn.ELU()
        self.num_classes = num_classes
        self.categorical_crossentropy = nn.CrossEntropyLoss()

    def bayesian_categorical_crossentropy_internal(self, pred_var, true):
        eps = 1e-6
        std = torch.sqrt(pred_var[:,self.num_classes])+ eps
        variance = pred_var[:, self.num_classes]
        variance_depressor = torch.exp(variance) - torch.ones_like(variance)
        pred = pred_var[:, 0:self.num_classes]
        undistorted_loss = self.categorical_crossentropy(pred+1e-15,true) #In pytorch loss (output,target)
        
        dist = tdist.Normal(torch.zeros_like(std), std)
        
        monte_carlo = [self.gaussian_categorical_crossentropy(pred, true, dist, undistorted_loss, self.num_classes) for _ in range(self.T)]
        monte_carlo = torch.stack(monte_carlo)
        variance_loss = torch.mean(monte_carlo,axis = 0) * undistorted_loss
        
        loss_final = variance_loss + undistorted_loss + variance_depressor
        return loss_final.mean()
    
    def gaussian_categorical_crossentropy(self, pred, true, dist, undistorted_loss, num_classes):
        std_samples = torch.squeeze(torch.transpose(dist.sample((num_classes,)), 0,1))
        distorted_loss = self.categorical_crossentropy(pred + std_samples, true)
        diff = undistorted_loss - distorted_loss
        return -1*self.ELU(diff)
    
    def forward(self, pred_var, true):
        return self.bayesian_categorical_crossentropy_internal(pred_var, true)

train_acc_history=[]
train_loss_history=[]
train_f1_history=[]

val_acc_history=[]
val_loss_history=[]
val_f1_history=[]

all_train_acc = []
all_train_loss = []
all_train_f1 = []
all_val_acc = []
all_val_loss = []
all_val_f1 = []

for fold_idx in range(10):
    print(f"Training and evaluating fold {fold_idx + 1}/10")
    
    train_dataset = data_dict[f'fold_{fold_idx + 1}']['train']
    val_dataset = data_dict[f'fold_{fold_idx + 1}']['val']
        
    device = torch.device('cuda:0')

    if model_name == "alexnet":
        """
        Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        aleatoric_model = AleatoricModel(model)

    elif model_name == "vgg":
        """
        VGG11_bn
        """
        model = models.vgg11_bn(pretrained=use_pretrained)
        # model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        aleatoric_model = AleatoricModel(model)

    elif model_name == "efficientnet":
        """
        EfficienNet
        """
        model = models.efficientnet_b0(use_pretrained)
        # model_ft = models.efficientnet_b0(use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        aleatoric_model = AleatoricModel(model)

    elif model_name == "convnext":
        """
        ConvNeXt
        """
        model = models.convnext_tiny(use_pretrained)
        # model_ft = models.convnext_tiny(use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        aleatoric_model = AleatoricModel(model)
    
    train_dataset = dataloaders_dict['train']
    val_dataset = dataloaders_dict['val']
    test_dataset = dataloaders_dict['test']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CUDA_LAUNCH_BLOCKING=1

    aleatoric_model.to(device)
    class_labels = []

    for _, c_label in train_dataset:
        class_labels += c_label

    label = []
    for c_label in class_labels:
        label.append(c_label.tolist())

    class_weights = compute_class_weight(
        class_weight = 'balanced', classes = np.unique(label), y = label)
    class_weights = torch.tensor(class_weights,dtype = torch.float)
    class_weights = class_weights.to(device)
    
    criterion_crossentropy =  nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    criterion_bayesian = bayesian_categorical_crossentropy(T_monte_carlo, 4)
    optimizer = optim.AdamW(aleatoric_model.parameters(), learning_rate)

    train_loss = 0.0
    train_variance_loss = 0.0
    train_crossentropy_loss =0.0
    val_loss = 0.0
    val_variance_loss = 0.0
    val_crossentropy_loss =0.0
    train_acc = 0.0
    val_acc = 0.0
    f1=0.0
    true_labels=[]
    pred_labels=[]
    count_train = 0.0
    count_var = 0.0
    best_acc = 0.0

    if mode =='Train':
        since = time.time()
        for epoch in tqdm(range(num_epochs)):
            aleatoric_model.train(True)
            for images,labels in train_dataset:
                images = images.to(device)
                labels = labels.to(device)
                count_train += len(labels)

                if len(images)==1 and batch_size > 1:
                    print(" Only one item in this batch ")
                    continue

                optimizer.zero_grad()
                logit_variance, logits, variance, variance_pre = aleatoric_model(images)
                _, preds = torch.max(logits,1)

                loss_crossentropy = criterion_crossentropy(logits,labels)
                loss_bayesian = criterion_bayesian(logit_variance, labels)
                loss = loss_div[0] * loss_crossentropy + loss_div[1] * loss_bayesian

                loss.backward()
                optimizer.step()

                train_loss += loss.data
                train_variance_loss += loss_bayesian
                train_crossentropy_loss += loss_crossentropy

                train_acc +=torch.sum(preds==labels.data)
                true_labels.extend(labels.data.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

            true_labelse = np.hstack(true_labels)
            pred_labelse = np.hstack(pred_labels)

            f1t = f1_score(true_labelse, pred_labelse, average='macro')
            avg_loss = train_loss / count_train
            avg_variance_loss = train_variance_loss / count_train
            avg_crossentropy_loss = train_crossentropy_loss / count_train
            avg_acc = train_acc / count_train
            print(" Total Loss: ",avg_loss," Variance Loss: ",avg_variance_loss, " Crossentropy Loss: ",avg_crossentropy_loss," Accuracy: ",avg_acc,"F1:",f1t)

            train_acc_history.append(avg_acc.cpu().detach().numpy())
            train_loss_history.append(avg_loss.cpu().detach().numpy())
            train_f1_history.append(f1t)


            #---------- Validation ------------#
            aleatoric_model.train(False)
            aleatoric_model.eval()
            true_labelsv=[]
            pred_labelsv=[]
            with torch.no_grad():
                for images,labels in val_dataset:
                    images = images.to(device)
                    labels = labels.to(device)
                    count_var += len(labels)

                    optimizer.zero_grad()
                    logit_variance, logits, variance, variance_pre = aleatoric_model(images)
                    _, preds = torch.max(logits,1)

                    loss_crossentropy = criterion_crossentropy(logits,labels)
                    loss_bayesian = criterion_bayesian(logit_variance, labels)
                    loss = 1.0 * loss_crossentropy + 0.2 * loss_bayesian

                    val_loss += loss.data
                    val_variance_loss += loss_bayesian
                    val_crossentropy_loss += loss_crossentropy

                    val_acc +=torch.sum(preds==labels.data)
                    
                    true_labelsv.extend(labels.data.cpu().numpy())
                    pred_labelsv.extend(preds.cpu().numpy())

            true_labelsve = np.hstack(true_labelsv)
            pred_labelsve = np.hstack(pred_labelsv)

            f1v = f1_score(true_labelsve, pred_labelsve, average='macro')

            avg_val_loss = val_loss / count_var
            avg_variance_loss = val_variance_loss / count_var
            avg_crossentropy_loss = val_crossentropy_loss / count_var
            avg_val_acc = val_acc / count_var

            
            val_acc_history.append(avg_val_acc.cpu().detach().numpy())
            val_loss_history.append(avg_val_loss.cpu().detach().numpy())
            val_f1_history.append(f1v)

            print(" Validation Loss: ",avg_val_loss," Validation Variance Loss: ",avg_variance_loss, " Validation Crossentropy Loss: ",avg_crossentropy_loss,"Validation Accuracy: ",avg_val_acc,"F1 Score:", f1v)

            if f1v > best_f1:
                best_f1= f1v
                best_epoch = epoch
                best_acc=val_acc
                report = classification_report(true_labelsve, pred_labelsve, labels=[0,1,2,3],zero_division=0)
                # print(best_f1)
                cf = confusion_matrix(true_labelsve, pred_labelsve)
                print("Saving this model at epoch --> ",epoch)
                model_save_path = "models_output_du"

                if not(os.path.exists(model_save_path)):
                    os.makedirs(model_save_path)
                torch.save(aleatoric_model, f'{model_save_path}/Model_origdata_lossratio_{loss_div[0]}_{loss_div[1]}_monte_carlo_{T_monte_carlo}_effnet_aleatoric_pytorch{model_name}______LR_{learning_rate}______Batch_{batch_size}______Epoch_{num_epochs}.pt')
            
            results_save_path = f"model_result_du/results_____Model_origdata_lossratio_{loss_div[0]}_{loss_div[1]}_monte_carlo_{T_monte_carlo}_effnet_aleatoric_pytorch{model_name}______LR_{learning_rate}______Batch_{batch_size}______Epoch_{num_epochs}/{model_name}_cropped"

            if not(os.path.exists(results_save_path)):
                os.makedirs(results_save_path)
            
            np.savetxt(f"{results_save_path}/train_acc_history.csv", train_acc_history, delimiter=",",fmt="%f")
            np.savetxt(f"{results_save_path}/train_loss_history.csv", train_loss_history, delimiter=",",fmt="%f")
            np.savetxt(f"{results_save_path}/train_f1_history.csv", train_f1_history, delimiter=",",fmt="%f")
            np.savetxt(f"{results_save_path}/val_acc_history.csv", val_acc_history, delimiter=",",fmt="%f")
            np.savetxt(f"{results_save_path}/val_loss_history.csv", val_loss_history, delimiter=",",fmt="%f")
            np.savetxt(f"{results_save_path}/val_f1_history.csv", val_f1_history, delimiter=",",fmt="%f")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best F1 score: {:.4f}'.format(best_f1))
        print('Best Epoch: '+ str(best_epoch))
        print(report)
        print(cf)
        report_path = f"model_result_du/results_____Model_origdata_lossratio_{loss_div[0]}_{loss_div[1]}_monte_carlo_{T_monte_carlo}_effnet_aleatoric_pytorch{model_name}______LR_{learning_rate}______Batch_{batch_size}______Epoch_{num_epochs}/{model_name}_cropped/classification_report.txt"
        with open(report_path, "w") as report_file:
        # report_file = open(, "w")
            report_file.write(report)
        report_file.close()

        cf_path = f"model_result_du/results_____Model_origdata_lossratio_{loss_div[0]}_{loss_div[1]}_monte_carlo_{T_monte_carlo}_effnet_aleatoric_pytorch{model_name}______LR_{learning_rate}______Batch_{batch_size}______Epoch_{num_epochs}/{model_name}_cropped/cf_report.csv"
        confusion_df = pd.DataFrame(cf)

        confusion_df.to_csv(cf_path)

    # After each fold, save the model and results
    model_save_path = f"fold_results/data_uncertainty/{model_name}/models_fold_{fold_idx + 1}"  # Adjust the path as needed
    results_save_path = f"fold_results/data_uncertainty/{model_name}/results_fold_{fold_idx + 1}"  # Adjust the path as needed
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    
    # Save the model
    torch.save(aleatoric_model, os.path.join(model_save_path, f'model_fold_{fold_idx + 1}.pt'))
    
    # Save the training and validation results
    np.savetxt(os.path.join(results_save_path, 'train_acc_history.csv'), train_acc_history, delimiter=",", fmt="%f")
    np.savetxt(os.path.join(results_save_path, 'train_loss_history.csv'), train_loss_history, delimiter=",", fmt="%f")
    np.savetxt(os.path.join(results_save_path, 'train_f1_history.csv'), train_f1_history, delimiter=",", fmt="%f")
    np.savetxt(os.path.join(results_save_path, 'val_acc_history.csv'), val_acc_history, delimiter=",", fmt="%f")
    np.savetxt(os.path.join(results_save_path, 'val_loss_history.csv'), val_loss_history, delimiter=",", fmt="%f")
    np.savetxt(os.path.join(results_save_path, 'val_f1_history.csv'), val_f1_history, delimiter=",", fmt="%f")
    
    # Append results to lists for aggregation
    all_train_acc.append(train_acc_history)
    all_train_loss.append(train_loss_history)
    all_train_f1.append(train_f1_history)
    all_val_acc.append(val_acc_history)
    all_val_loss.append(val_loss_history)
    all_val_f1.append(val_f1_history)

    confusion_df = pd.DataFrame(cf)
    file_fold_path = os.path.join(results_save_path, 'classification_report.txt')

    with open(file_fold_path, "w") as report_file:
            report_file.write(report)
