import torch.nn as nn
#%%
def classifier_layers():
   cnn = nn.Sequential(

      nn.Conv2d(1, 10, kernel_size=3),  
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(10, 10, kernel_size=3),  
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Flatten(),

      nn.Linear(250, 2),
    )
   return cnn

def math_layers():
   cnn = nn.Sequential(

      nn.Conv2d(1, 10, kernel_size=3),  
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(10, 10, kernel_size=3),  
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Flatten(),

      nn.Linear(250, 2),  
    )
   return cnn

def mnist_layers():
   cnn = nn.Sequential(

      nn.Conv2d(1, 10, kernel_size=3),  
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(10, 10, kernel_size=3),  
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Flatten(),

      nn.Linear(250, 10),  
    )
   return cnn