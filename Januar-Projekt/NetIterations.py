import torch.nn as nn

def cnn_base():
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

def cnn_addedlayers():
    cnn = nn.Sequential(
       
        nn.Conv2d(1, 50, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(50, 40, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(40, 30, kernel_size=3),
        nn.ReLU(),

        nn.Conv2d(30, 20, kernel_size=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(20,10),
    )
    return cnn

def cnn_base_dropout():
   cnn = nn.Sequential(
      nn.Conv2d(1, 10, kernel_size=3),  
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout(p=0.2),
      nn.ReLU(),

      nn.Conv2d(10, 10, kernel_size=3),  
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout(p=0.2),
      nn.ReLU(),

      nn.Flatten(),

      nn.Linear(250, 10),
   )
   return cnn

def cnn_added_layers_dropout():
    cnn = nn.Sequential(
       
        nn.Conv2d(1, 50, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.2),

        nn.Conv2d(50, 40, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.2),

        nn.Conv2d(40, 30, kernel_size=3),
        nn.ReLU(),
        nn.Dropout(p=0.2),

        nn.Conv2d(30, 20, kernel_size=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.2),

        nn.Flatten(),

        nn.Linear(20,10),
    )
    return cnn