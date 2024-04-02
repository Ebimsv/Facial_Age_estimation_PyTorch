import torch.cuda

lr = 0.0005
wd = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 1
train_batch_size = 100
valid_batch_size = 128
