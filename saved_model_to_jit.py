import torch
from model import AgeEstimationModel

model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2')
model.load_state_dict(torch.load("checkpoints/epoch:28-loss_valid:5.95.pt"))
model.to('cuda')
model.eval()
model_jit = torch.jit.script(model)
torch.jit.save(model_jit, 'model-jit.pt')
