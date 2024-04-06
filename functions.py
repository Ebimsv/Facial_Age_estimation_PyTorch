import torch
from utils import AverageMeter
from tqdm import tqdm

from config import config


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=1):
    model.train()
    loss_train = AverageMeter()
    
    metric.reset()
    with tqdm(train_loader, unit="batch", desc=f'Epoch: {epoch+1}/{config["epochs"]}', bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}') as tqdm_dataloader:
        
        for inputs, targets, _, _ in tqdm_dataloader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # By setting n to len(targets), we ensure, that the loss is accurately calculated and updated, regardless of any changes in batch size.
            loss_train.update(loss.item(), n=len(targets))
            metric.update(outputs, targets)

            tqdm_dataloader.set_postfix(loss=loss_train.avg, metric=metric.compute().item())
        
        del outputs
        torch.cuda.empty_cache()
        return model, loss_train.avg, metric.compute().item()


def validation(model, valid_loader, loss_fn, metric):
    model.eval()
    loss_valid = AverageMeter()

    metric.reset()

    with tqdm(valid_loader, unit="batch", desc=f'Evaluating... ', 
              bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}') as tqdm_dataloader:

        with torch.no_grad():
            for i, (inputs, targets, _, _) in enumerate(tqdm_dataloader):
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])

                outputs = model(inputs)

                loss = loss_fn(outputs, targets)

                # `n=len(targets)` for Dynamic Batch Size Consideration
                loss_valid.update(loss.item(), n=len(targets))
                metric.update(outputs, targets)
                
    del outputs
    torch.cuda.empty_cache()
    return loss_valid.avg, metric.compute().item()
