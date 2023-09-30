import torch.cuda

lr = 0.0005
wd = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 10

