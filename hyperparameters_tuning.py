from torch import nn
from torch.utils.data import DataLoader, random_split
from torch import optim
from model import model, device
from dataset_dataloader import train_set, train_loader
from functions import train_one_epoch
from model import AgeEstimationModel
from prettytable import PrettyTable

model = model.model
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Finding best Hyper-parameters
# Step 1: Calculate the loss for an untrained model using a few batches.
x_batch, y_batch, _, _ = next(iter(train_loader))
outputs = model(x_batch.to(device))
loss = loss_fn(outputs, y_batch.unsqueeze(1).to(device))
# print(loss)

# Step 2: Try to train and Overfit the model on a small subset of the dataset.
_, mini_train_dataset = random_split(train_set, (len(train_set)-1000, 1000))
mini_train_loader = DataLoader(mini_train_dataset, 5)

num_epochs = 5
# for epoch in range(num_epochs):
#     model, loss_train = train_one_epoch(model, mini_train_loader, loss_fn, optimizer, metric='mse', epoch=epoch)

print('')

# Step 3: Train the model for a limited number of epochs for all data, experimenting with various learning rates.
for lr in [0.001, 0.0001, 0.0005]:
    print(f'lr is: {lr}')
    model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_epochs):
        model, loss_train = train_one_epoch(model, train_loader, loss_fn, optimizer, metric='mae', epoch=epoch)
    print('')

# Step 4: Create a small grid using the weight decay and the best learning rate.
small_grid_list = []
for lr in [0.001, 0.0001, 0.0005]:
    for wd in [1e-4, 1e-5, 0.]:
        print(f'LR={lr}, WD={wd}')
        model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
        loss_fn = nn.L1Loss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for epoch in range(num_epochs):
            model, loss_train = train_one_epoch(model, train_loader, loss_fn, optimizer, metric='mae', epoch=epoch)

        small_grid_list.append([lr, wd, loss_train])


table = PrettyTable(['LR', 'WD', 'Loss', 'Accuracy'])

for rec in small_grid_list:
    table.add_row(rec)

print(table)
