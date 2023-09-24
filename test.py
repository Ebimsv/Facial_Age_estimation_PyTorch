import torch
from torch import nn, optim

from dataset_dataloader import test_loader
import config
from utils import AverageMeter
from model import model

loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.wd)

loss_test = AverageMeter()
for images, targets, _, _ in test_loader:
    images = images.to(config.device)
    labels = targets.to(config.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        loss = loss_fn(outputs, labels.unsqueeze(1).float().to(config.device))
        loss_test.update(loss.item())
    print(loss_test.avg)

# Inference:
"""
- Write an inference function.
- load an image from outside the UTKFace dataset
- and evaluate the model's prediction.
"""