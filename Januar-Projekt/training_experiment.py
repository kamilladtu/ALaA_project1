# %%
import NetIterations
import NetUtils
import torch.nn as nn
# %%
net1 = NetIterations.cnn_base()
net2 = NetIterations.cnn_base_dropout()
net3 = NetIterations.cnn_addedlayers()
net4 = NetIterations.cnn_added_layers_dropout()
loss_fn = nn.CrossEntropyLoss()

#%%
training_dataloader = NetUtils.get_dataloader("MNIST", train=True, batch_size=5000)
testing_dataloader = NetUtils.get_dataloader("MNIST", train=False, batch_size=1000)

# %%
model1 = NetUtils.NeuralNet(net1, loss_fn, training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)
model2 = NetUtils.NeuralNet(net2, loss_fn, training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)
model3 = NetUtils.NeuralNet(net3, loss_fn, training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)
model4 = NetUtils.NeuralNet(net4, loss_fn, training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)

# %%
epochs = 40
print("MODEL 1")
print("----------------------------------------")
model1.train_test_loop(epochs, print_info=True, save=True, savefolder='weights/model1')
print("MODEL 2")
print("----------------------------------------")
model2.train_test_loop(epochs, print_info=True, save=True, savefolder='weights/model2')
print("MODEL 3")
print("----------------------------------------")
model3.train_test_loop(epochs, print_info=True, save=True, savefolder='weights/model3')
print("MODEL 4")
print("----------------------------------------")
model4.train_test_loop(epochs, print_info=True, save=True, savefolder='weights/model4')
