from torch import nn, optim

import config
from model import AgeEstimationModel
import matplotlib.pyplot as plt

from functions import train_one_epoch, validation
from dataset_dataloader import train_loader, valid_loader
from config import *
from utils import AverageMeter

# Step 5: Train model for longer epochs using the best model from step 4 in hyperparameters_tuning.
# Define model, define optimizer and Set learning rate and weight decay.
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.wd)

# Write code to train the model for num_epochs epochs.
best_model = torch.inf
loss_train_hist = []
loss_valid_hist = []

acc_train_hist = []
acc_valid_hist = []

num_epochs = 10

for epoch in range(num_epochs):
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

    if epoch % 10 == 0:
        print()
        print(f'Train: loss={loss_train:.3}')
        print(f'Valid: loss={loss_valid:.3}')


# Plots: Plot learning curves
plt.plot(range(num_epochs), loss_train_hist, 'r-', label='Train')
plt.plot(range(num_epochs), loss_valid_hist, 'b-', label='Validation')

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()

# Test: Test your model using data from the test set and images that are not present in the dataset.
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
model.eval()
loss_fn = nn.L1Loss()
loss_test = AverageMeter()
