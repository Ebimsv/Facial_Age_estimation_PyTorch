import csv
import os
import pandas as pd

# before creating csv, I want to find all files extensions with the below code:
def get_file_extensions(folder_path):
    file_extensions = set([os.path.splitext(filename)[1] for filename in os.listdir(folder_path)])
    return file_extensions

folder_path = '/home/ebrahim/Python_projects/projects/dataset/UTKface/' 
extensions = get_file_extensions(folder_path)
print(extensions)

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
            if gender or age or ethnicity == '':
                print('hi')
                continue
            gender = 'Male' if int(gender) == 0 else 'Female'
            if ethnicity != str:
                ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others'][int(ethnicity)]

            if int(age) < 85:
                data = [image_file, age, ethnicity, gender]
                writer.writerow(data)


# Replace with the actual path to your UTK dataset images folder
dataset_folder = '/home/ebrahim/Python_projects/projects/dataset/UTKface/'
create_csv(dataset_folder)

df = pd.read_csv('./csv_dataset/utkface_dataset.csv')
print(f'Dataframe length: {len(df)}')
print()
print(df.head())