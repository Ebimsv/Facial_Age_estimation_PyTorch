import torch
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from model import AgeEstimationModel

import torchvision.transforms as transforms

from config import config

    
def inference(model, image_path, output_path):
    model.eval() 
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([T.Resize(((config['img_width'], config['img_height']))),
                               T.ToTensor(),
                               T.Normalize(mean=config['mean'], 
                                           std=config['std'])
                              ])
        input_data = transform(image).unsqueeze(0).to(config['device']) 
        outputs = model(input_data)  # Forward pass through the model
        
        # Extract the age estimation value from the output tensor
        age_estimation = outputs.item()
        
        # Create a PIL image with the age estimation value as text
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        text = f"Age: {age_estimation:.2f}"  
        draw.text((10, 10), text, fill=(255, 0, 0))
        
        # Save the output image
        output_image.save(output_path)

path = "/home/deep/projects/Mousavi/Facial_Age_estimation_PyTorch/checkpoints/"
checkpoint_files = glob.glob(os.path.join(path, 'epoch-*-loss_valid-*.pt'))
latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
# Load the model using the latest checkpoint file
model.load_state_dict(torch.load(latest_checkpoint))

image_path_test = config['image_path_test'] # Path to the input image
output_path_test = config['output_path_test']  # Path to save the output image
inference(model, image_path_test, output_path_test)
