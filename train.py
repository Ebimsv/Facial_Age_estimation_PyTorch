from torch import nn, optim

import config
from model import AgeEstimationModel
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import os
from functions import train_one_epoch, validation
from custom_dataset_dataloader import train_loader, valid_loader
from config import *

# create a folder to save checkpoints
path = 'checkpoints'
try:
    os.mkdir(path)
except OSError as error:
    print(error)

# Step 5: Train model for longer epochs using the best model from step 4 in hyperparameters_tuning.
# Define model, define optimizer and Set learning rate and weight decay.
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.wd)

# Write code to train the model for num_epochs epochs.
best_loss = torch.inf
before_model_path = None

loss_train_hist = []
loss_valid_hist = []

acc_train_hist = []
acc_valid_hist = []

writer = SummaryWriter()


for epoch in range(config.num_epochs):
    # Train
    model, loss_train = train_one_epoch(model,
                                        train_loader,
                                        loss_fn,
                                        optimizer,
                                        epoch)
    # Validation
    loss_valid = validation(model,
                            valid_loader,
                            loss_fn)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    if loss_valid < best_loss:
        best_loss = loss_valid
        if before_model_path is not None:
            os.remove(before_model_path)
        before_model_path = f'checkpoints/epoch-{epoch}-loss_valid-{best_loss:.3}.pt'
        torch.save(model.state_dict(), before_model_path)
        print(f'\nModel saved in epoch: {epoch}')

    writer.add_scalar('Loss/train', loss_train, epoch)
    writer.add_scalar('Loss/test', loss_valid, epoch)
    
    if epoch % 5 == 0:
        print()
        print(f'Train: loss={loss_train:.3}')
        print(f'Valid: loss={loss_valid:.3}')


writer.close()

# Plots: Plot learning curves
plt.plot(range(num_epochs), loss_train_hist, 'r-', label='Train')
plt.plot(range(num_epochs), loss_valid_hist, 'b-', label='Validation')

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()


# Test: Test your model using data from the test set and images that are not present in the dataset.
# model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
# model.eval()
# loss_fn = nn.L1Loss()
# loss_test = AverageMeter()
