# %%
import NetUtils
import cnn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn as nn

class UniformTargetDataset(Dataset):
    def __init__(self, original_dataset, new_target):
        self.original_dataset = original_dataset
        self.new_target = new_target

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, _ = self.original_dataset[idx]
        return image, self.new_target

# %%

# MNIST and math dataset
mnist_trainingdata = NetUtils.get_dataset("MNIST", train=True)
mnist_testingdata = NetUtils.get_dataset("MNIST", train=False)
math_trainingdata = NetUtils.get_dataset("MATH", train=True)
math_testingdata = NetUtils.get_dataset("MATH", train=False)

# Uniform targets
uniform_mnist_trainingdata = UniformTargetDataset(mnist_trainingdata, 0)
uniform_mnist_testingdata = UniformTargetDataset(mnist_testingdata, 0)
uniform_math_trainingdata = UniformTargetDataset(math_trainingdata, 1)
uniform_math_testingdata = UniformTargetDataset(math_testingdata, 1)

# %%
# Classifier
classifier_trainingdata = ConcatDataset([uniform_mnist_trainingdata, uniform_math_trainingdata])
classifier_testingdata = ConcatDataset([uniform_mnist_testingdata, uniform_math_testingdata])
classifier_trainingloader = DataLoader(dataset=classifier_trainingdata, shuffle=True, batch_size=5000)
classifier_testingloader = DataLoader(dataset=classifier_testingdata, shuffle=True, batch_size=5000)

# %%
math_trainingloader = NetUtils.get_dataloader("MATH", train=True, batch_size=5000)
mnist_trainingloader = NetUtils.get_dataloader("MNIST", train=True, batch_size=5000)
math_testingloader = NetUtils.get_dataloader("MATH", train=False, batch_size=5000)
mnist_testingloader = NetUtils.get_dataloader("MNIST", train=False, batch_size=5000)

# %%
loss_fn = nn.CrossEntropyLoss()

classifier_layers = cnn.classifier_layers()
mnist_layers = cnn.mnist_layers()
math_layers = cnn.math_layers()


classifier_net = NetUtils.NeuralNet(classifier_layers, loss_fn, training_dataloader=classifier_trainingloader, testing_dataloader=classifier_trainingloader)
mnist_net = NetUtils.NeuralNet(mnist_layers, loss_fn, training_dataloader=mnist_trainingloader, testing_dataloader=mnist_trainingloader)
math_net = NetUtils.NeuralNet(math_layers, loss_fn, training_dataloader=math_trainingloader, testing_dataloader=math_testingloader)

# %%
epochs = 40
print("CLASSIFIER")
print("----------------------------------------")
classifier_net.train_test_loop(epochs, print_info=True, save=True, savefolder='weights/classifier')
print("MNIST")
print("----------------------------------------")
mnist_net.train_test_loop(epochs, print_info=True, save=True, savefolder='weights/mnist')
print("MATH")
print("----------------------------------------")
math_net.train_test_loop(epochs, print_info=True, save=True, savefolder='weights/math')
