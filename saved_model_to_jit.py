import timm
import torch
from train import ImageTaggerModel


model = ImageTaggerModel(pretrained=True, num_classes=2)
model.load_state_dict(torch.load("epoch:28-acc_valid:0.9588607549667358-acc_train:0.9734758734703064.pt"))
model.to('cuda')
model.eval()
model_jit = torch.jit.script(model)
torch.jit.save(model_jit, 'model_jit.pt')
