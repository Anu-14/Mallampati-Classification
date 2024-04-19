import torch
# from torch import nn, save, load
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize
# from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
# import os
# from math import *


model_names = ["alexnet", "vgg", "squeezenet", "efficientnet", "convnext"]
modes = ["normal", "avg", "sum", "res_avg", "res_sum"]
folds = []

for i in range(1, 11):
    folds.append(i)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paths = []

for mode in modes:
    for model_name in model_names:
        for fold in folds:
            model_paths.append(f'Results/mallampati_data/{model_name}/{mode}/fold_{fold}/Model_{model_name}_LR_0.0001_Batch_16_Epoch_100.pt')

def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    true = []
    pred = []
    image_paths = []
    df = pd.DataFrame()
    with torch.no_grad():
        for img,labels in test_loader:
            images, labels = img.to(device),labels.to(device)
            # run the model on the test set to predict labels
            outputs = []
            for _ in range(5):
                outputs.append(model(images))
                # the label with the highest energy will be our prediction


            final_output = sum(outputs) / len(outputs)
            _, predicted = torch.max(final_output.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            true.extend(labels.cpu().numpy())
            pred.extend(predicted.cpu().numpy())
            image_paths.extend([path for path in img])
    true = np.array(true)
    pred = np.array(pred)
    df = pd.DataFrame({'image_path': image_paths,
                'true_label': true,
                'predicted_label': pred})

    cf = confusion_matrix(true, pred)
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    print("-------------------Test Metrics--------------------")
    print(model_path)
    print("---------------------------------------------------")
    print("Accuracy:" + str(accuracy))
    print("F1 Score:" + str(f1_score(true, pred, average='macro') * 100))
    # print(cf)
    print()


def testBatch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                            for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                            for j in range(batch_size)))

for model_path in model_paths:
    # try:
    model = torch.load(model_path)
    model.to(device)

    transform = transforms.Compose(
        [Resize([224,224]),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # Same as for your validation data, e.g. Resize, ToTensor, Normalize, ...


    test_data = datasets.ImageFolder('datasets/mallampati_data/test', transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)

    testAccuracy()
    print()
    print("---------------------------------------------------")
    # except:
    #     print(f"Not printing for {model_path}")
