# %%
import os
import cv2
import ImageProcessing as ip
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn as nn
import time

MNIST_FOLDER = "data/MNIST"
MATH_FOLDER = "data/symbols_extended"
SPLITKEY = 7

# %%
class NeuralNet():
    def __init__(self, model, loss_fn, training_dataloader=None, testing_dataloader=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, maximize=False)
        self.training_dataloader = training_dataloader
        self.testing_dataloader = testing_dataloader
        self.softmax_fn = nn.Softmax()

    def training_loop(self, print_info=False):
            total_loss = 0
            size = len(self.training_dataloader)
            accuracies = []
            
            for images,labels in self.training_dataloader:

                correct_guesses = 0
                batch_size = len(images)

                self.optimizer.zero_grad()
                output = self.model(images)
                
                for idx, guess in enumerate(output):
                    try:
                        if guess.argmax() == labels[idx]:
                            correct_guesses += 1
                    except:
                        print(guess)
                        print(output)

                loss = self.loss_fn(output, labels)
                total_loss += loss
                accuracies.append(correct_guesses / batch_size)
                
                loss.backward()
                self.optimizer.step()
            accuracies = np.array(accuracies)
            avg_loss = total_loss / size
            avg_accuracy = accuracies.mean()

            if print_info:
                print("Training")
                print(f"Total loss: {total_loss}")
                print(f"Avg loss: {avg_loss}")
                print(f"Accuracy : {avg_accuracy}")
                print()
            return avg_loss, avg_accuracy

    
    def testing_loop(self, print_info=False):
            total_loss = 0
            size = len(self.testing_dataloader)
            accuracies = []
            
            with torch.no_grad():
                for images,labels in self.testing_dataloader:
                    correct_guesses = 0
                    batch_size = len(images)
                    output = self.model(images)
            
                    for idx, guess in enumerate(output):
                        if guess.argmax() == labels[idx]:
                            correct_guesses += 1


                    accuracy = correct_guesses / batch_size
                    accuracies.append(accuracy)
                    loss = self.loss_fn(output, labels)
                    total_loss += loss
            accuracies = np.array(accuracies)
            
            avg_loss = total_loss / size
            avg_accuracy = accuracies.mean()
            if print_info:
                print(f"Testing")
                print(f"Total Loss: {total_loss}")
                print(f"Avg Loss: {avg_loss}")
                print(f"Accuracy : {avg_accuracy}")
                print("--------------")

            return avg_loss, avg_accuracy

    def train_test_loop(self, epochs, print_info=False, save=False, savefolder=''):
        total_time = 0
        for epoch in range(epochs):
            start_time = time.time()

            if print_info:
                print(f"Epoch: {epoch + 1}")
                print(f"Total time spent: {total_time:.2f}")
                avg_time = total_time / (epoch + 1)
                print(f"Seconds pr. epoch: {avg_time:.2f}")
                print(f"Time remaining: {((epochs - epoch - 1) * avg_time):.2f}")
                print()

            self.training_loop(print_info=print_info)
            testing_loss, testing_accuracy = self.testing_loop(print_info=print_info)
            total_time += (time.time() - start_time)

            
            if save:
                path = f"{savefolder}/ep{epoch}loss{testing_loss:.6f}acc{testing_accuracy:.6f}"
                torch.save(self.model.state_dict(), path)




# %%
def get_dataloader(dataset, batch_size=1, train=True):
    return DataLoader(get_dataset(dataset, train), batch_size=batch_size, shuffle=True)

# Tager input 
def get_dataset(dataset, train=True):
    if dataset =="MATH":
        trainingset, testset = load_math_dataset()
        if train:
            images, labels = zip(*trainingset)
            images, labels = format_data_for_model(images, labels)
            return TensorDataset(images, labels)
        else:
            images, labels = zip(*testset)
            images, labels = format_data_for_model(images, labels)
            return TensorDataset(images, labels)

    elif dataset == "MNIST":
        return  MNIST(root='data', transform=ToTensor(), train=train)

def load_math_dataset():
    images, labels = load_math_images(MATH_FOLDER)
    trainingset, testset = get_subset(images, labels, SPLITKEY)
    return trainingset, testset
 
def get_subset(images, labels, splitkey):
    testset = []
    trainingset = []

    data_list= list(zip(images, labels))

    # https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
    sorted_list = sorted(data_list, key=lambda x: x[1])

    for idx, i in enumerate(sorted_list):
        if idx % splitkey == 0: 
            testset.append(i)
        else:
            trainingset.append(i)

    return trainingset, testset


def format_data_for_model(images, labels):
    images = [ip.preprocess_stack(image) for image in images]

    for idx, image in enumerate(images):
        if image.shape != (28, 28):
            images[idx] = cv2.resize(image, (28,28))   

    images, labels = listdata_to_tensor(images, labels)
    return images, labels

def listdata_to_tensor(images, labels):
    labels = np.array(labels)
    images = np.array(images)
    labels = torch.tensor(labels)
    images = torch.tensor(images)
    images = images.float()/255
    images = images.unsqueeze(1)

    return images, labels

def load_math_images(foldername):
    images = []
    labels = []
    
    folders = os.listdir(foldername)
    if ".DS_Store" in folders:
        folders.remove(".DS_Store")

    for folder in folders:
        file_paths = os.listdir(f"{foldername}/{folder}")
        if ".DS_Store" in file_paths:
            file_paths.remove(".DS_Store")

        for path in file_paths:
            abs_path = f"{foldername}/{folder}/{path}"
            images.append(cv2.imread(abs_path))
            if folder == 'plus':
                labels.append(0)
            elif folder == 'minus':
                labels.append(1)
        
    return images, labels
