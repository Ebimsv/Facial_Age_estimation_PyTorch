import torch
from utils import AverageMeter
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric='mse', epoch=1):
    model.train()
    loss_train = AverageMeter()
    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets, _, _ in tepoch:
            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets.unsqueeze(1).float().to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item())
            tepoch.set_postfix(loss=loss_train.avg)
        return model, loss_train.avg


def validation(model, valid_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        loss_valid = AverageMeter()
        for i, (inputs, targets, _, _) in enumerate(valid_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets.unsqueeze(1).float().to(device))
            loss_valid.update(loss.item())

    return loss_valid.avg
