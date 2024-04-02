import torch
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from model import AgeEstimationModel

import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
def inference(model, image_path, output_path):
    model.eval() 
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([T.Resize((128, 128)),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                              ])
        input_data = transform(image).unsqueeze(0).to(device) 
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
checkpoint_path = os.path.join(path, 'epoch-0-loss_valid-9.72.pt')  # Path to the saved checkpoint file
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
model.load_state_dict(torch.load(checkpoint_path))

image_path = '/home/deep/projects/Mousavi/Facial_Age_estimation_PyTorch/img_test/30_1_2.jpg'  # Path to the input image
output_path = '/home/deep/projects/Mousavi/Facial_Age_estimation_PyTorch/img_test/output.jpg'  # Path to save the output image
inference(model, image_path, output_path)
