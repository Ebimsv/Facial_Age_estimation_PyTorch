import torch
from torch import nn
from torchvision.models import resnet, efficientnet_b0
import timm


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

        elif model_name == 'efficientnet':
            self.model = efficientnet_b0()
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=256, bias=True),
                nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))
            nn.init.xavier_uniform_(self.model.classifier[2].weight)
            nn.init.zeros_(self.model.classifier[2].bias)

        elif model_name == 'vit':
            self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=pretrain_weights)
            num_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Linear(256, self.output_nodes))

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        x = self.model(x)
        return x


# model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2')
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='vit', pretrain_weights=True)

def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


if __name__ == '__main__':
    print(num_trainable_params(model))
