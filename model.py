import torch
from torch import nn
from torchvision.models import resnet, efficientnet_b0


# Custom Model
class AgeEstimationModel(nn.Module):

    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(AgeEstimationModel, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=2048, out_features=256, bias=True),
                nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))
        else:
            self.model = efficientnet_b0()
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=256, bias=True),
                nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))
            nn.init.xavier_uniform_(self.model.classifier[2].weight)
            nn.init.zeros_(self.model.classifier[2].bias)

    def forward(self, x):
        x = self.model(x)
        return x


model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2')


def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums


print(num_trainable_params(model))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
