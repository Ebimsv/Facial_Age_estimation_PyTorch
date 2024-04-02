import torch.cuda

lr = 0.0005
wd = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 30
train_batch_size = 64
valid_batch_size = 128
