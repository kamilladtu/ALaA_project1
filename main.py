import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
import random
from keras.datasets import mnist


#loading mnist data:
(train_X, train_y), (test_X, test_y) = mnist.load_data()



# Shape:

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
