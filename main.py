# import libraries
import numpy as np
import pandas as pd
import os
import random
import time
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from CatDogDataset import *
from CatDogCNN import *
from CatDogTest import *



# import data
img_files = os.listdir('train/')
img_files = list(filter(lambda x: x != 'train', img_files))
def train_path(p): return f"train/{p}"
img_files = list(map(train_path, img_files))

print("total training images", len(img_files))
print("First item", img_files[0])

# separate into train and test sets
random.shuffle(img_files)
train = img_files[:20000]
test = img_files[20000:]
print("train size", len(train))
print("test size", len(test))


# image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# create train dataset
train_ds = CatDogDataset(train, transform)
train_dl = DataLoader(train_ds, batch_size=100)
print(len(train_ds), len(train_dl))

# create test dataset
test_ds = CatDogDataset(test, transform)
test_dl = DataLoader(test_ds, batch_size=100)
print(len(test_ds), len(test_dl))

# Create instance of the model
model = CatAndDogConvNet()

losses = []
accuracies = []
epoches = 8
start = time.time()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Model Training...
for epoch in range(epoches):

    epoch_loss = 0
    epoch_accuracy = 0

    for X, y in train_dl:

        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = ((preds.argmax(dim=1) == y).float().mean())
        epoch_accuracy += accuracy
        epoch_loss += loss
        print('.', end='', flush=True)

    epoch_accuracy = epoch_accuracy/len(train_dl)
    accuracies.append(epoch_accuracy)
    epoch_loss = epoch_loss / len(train_dl)
    losses.append(epoch_loss)

    print("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))

    # test set accuracy
    with torch.no_grad():

        test_epoch_loss = 0
        test_epoch_accuracy = 0

        for test_X, test_y in test_dl:

            test_preds = model(test_X)
            test_loss = loss_fn(test_preds, test_y)

            test_epoch_loss += test_loss            
            test_accuracy = ((test_preds.argmax(dim=1) == test_y).float().mean())
            test_epoch_accuracy += test_accuracy

        test_epoch_accuracy = test_epoch_accuracy/len(test_dl)
        test_epoch_loss = test_epoch_loss / len(test_dl)

        print("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch, test_epoch_loss, test_epoch_accuracy, time.time() - start))


# prepare test dataset
test_files = os.listdir('test/')
test_files = list(filter(lambda x: x != 'test', test_files))
def test_path(p): return f"test/{p}"
test_files = list(map(test_path, test_files))

dog_probs = []

with torch.no_grad():
    for X, fileid in test_dl:
        preds = model(X)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))
        
# display some images
for img, probs in zip(test_files[:5], dog_probs[:5]):
    print("Showing image...")
    pil_im = Image.open(img, 'r')
    label = "dog" if probs[1] > 0.5 else "cat"
    title = "prob of dog: " + str(probs[1]) + " Classified as: " + label
    plt.figure()
    plt.imshow(pil_im)
    plt.suptitle(title)
    plt.show()