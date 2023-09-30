import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import config


# Create a csv file which contains labels
def create_csv(dataset_folder):
    image_files = os.listdir(dataset_folder)
    header = ['image_name', 'age', 'ethnicity', 'gender']
    with open('csv_dataset/utkface_dataset.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, image_file in enumerate(image_files):
            if len(image_file.split('_')) < 4:
                continue
            age, gender, ethnicity = image_file.split('_')[:3]
            if gender == '':
                continue
            gender = 'Male' if int(gender) == 0 else 'Female'
            if ethnicity != str:
                ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(ethnicity)]

            data = [image_file, age, ethnicity, gender]
            writer.writerow(data)


# Replace with the actual path to your UTK dataset images folder
dataset_folder = '/home/ebrahim/Python_projects/projects/dataset/UTKface/'
create_csv(dataset_folder)

df = pd.read_csv('./csv_dataset/utkface_dataset.csv')
df.head()

df_train, df_test = train_test_split(df, train_size=0.8, random_state=42)
df_train, df_valid = train_test_split(df_train, train_size=0.85, random_state=42)

# Save the training, validation, and test sets in separate CSV files.
df_train.to_csv('./csv_dataset/train_set.csv', index=False)
df_valid.to_csv('./csv_dataset/valid_set.csv', index=False)
df_test.to_csv('./csv_dataset/test_set.csv', index=False)

print('All CSV files created successfully.')

# Define transformations
transform_train = T.Compose([T.Resize((128, 128)),
                             T.RandomHorizontalFlip(),
                             T.RandomRotation(degrees=15),
                             T.ColorJitter(brightness=(0.5, 1.5), contrast=1, saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])

transform_test = T.Compose([T.Resize((128, 128)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])


# Custom dataset: A custom dataset class for UTKFace.
class UTKDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_row = self.data.iloc[idx].values
        img_dir = os.path.join(self.root_dir, one_row[0])

        image = Image.open(img_dir).convert('RGB')
        image = self.transform(image)
        age = torch.tensor(one_row[1])

        gender = one_row[2]
        ethnicity = one_row[3]

        return image, age, gender, ethnicity


# Utilize the UTKDataset class  to instantiate dataset objects for the training, validation, and test sets.
root_dir = '/home/ebrahim/Python_projects/projects/dataset/UTKface/'
csv_file_train = 'csv_dataset/train_set.csv'
csv_file_valid = 'csv_dataset/valid_set.csv'
csv_file_test = 'csv_dataset/test_set.csv'

# Define dataloader: Write dataloaders for the training, validation, and test sets.
train_set = UTKDataset(root_dir, csv_file_train, transform_train)
valid_set = UTKDataset(root_dir, csv_file_valid, transform_test)
test_set = UTKDataset(root_dir, csv_file_test, transform_test)

train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=config.valid_batch_size)
test_loader = DataLoader(test_set, batch_size=config.valid_batch_size)

# Test the dataloaders using next(iter())
batch_train = next(iter(train_loader))
print(f'img shape is: {batch_train[0].shape},\nages are {len(batch_train[1])},\nethnicities are {len(batch_train[2])}, '
      f'\ngender is {len(batch_train[3])}')
