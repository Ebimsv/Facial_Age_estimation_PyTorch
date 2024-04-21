![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/Age-Estimation.png)

# About This Project
This repository provides a detailed, step-by-step implementation guide for facial age estimation using PyTorch, a popular deep learning framework. The goal of this project is to accurately estimate the age of individuals based on their facial appearance.

# Step 1: Accurate and concise definition of the problem
Age estimation refers to the process of estimating a person's age based on various observable characteristics or features. It is often performed using computer vision techniques that analyze facial attributes such as wrinkles, skin texture, and hair color. By comparing these characteristics with a database of known age examples, algorithms can make an educated guess about a person's age. **However, it is important to note that age estimation is not always accurate and can be affected by factors such as lighting conditions, facial expressions, race, genetics, and individual variations such as makeup**. It is primarily used in applications like biometrics, demographic analysis, or age-restricted access control systems.

## The goal of solving a problem or challenge
<details>
<summary><b>Here are some specific goals that researchers and developers in this field aim to achieve:
</b></summary><br/>

- **Improved Accuracy**  
This involves reducing the margin of error in estimating a person's age and increasing the precision of the predictions. The ultimate aim is to develop models that can estimate age with a high level of accuracy, approaching or surpassing human-level performance.

- **Reduced Bias**  
Biases can arise due to factors such as ethnicity, gender, or other demographic characteristics. It is crucial to develop models that are fair and unbiased, providing accurate age estimates regardless of an individual's background or appearance.

- **Robustness**  
Facial age estimation algorithms should be robust and reliable under various conditions and scenarios. They should be able to handle variations in lighting, pose, expression, and other factors that may affect facial appearance. Robustness ensures that age estimation models perform consistently and accurately across different environments and datasets.

- **Generalization**  
Age estimation algorithms should be trained on diverse datasets representing different populations, age groups, and ethnicities. The goal is to develop models that can accurately estimate age for individuals from various demographics and cultural backgrounds, rather than being limited to a specific subset of the population.

- **Real-Time Performance**  
In many applications, such as surveillance systems, age estimation algorithms need to operate in real-time. Therefore, a goal is to develop efficient algorithms that can provide age estimates quickly, allowing for practical implementation in real-world scenarios without significant computational overhead.

- **Cross-Domain Adaptability**  
Facial age estimation algorithms should be adaptable to different domains and applications. For example, they should be effective in estimating age from photographs, video frames, or even in live video streams. The ability to adapt to different data sources and domains expands the potential applications of age estimation technology.

- **Privacy and Ethical Considerations**  
The development of age estimation algorithms should take into account data protection, consent, and potential misuse. Ensuring that privacy is respected and ethical guidelines are followed is a crucial goal in this field.

By striving to achieve these goals, researchers and developers aim to advance the field of facial age estimation and create algorithms that can have practical applications in areas such as biometrics, demographics analysis, age-specific marketing, and personalized user experiences.

</details>
  
# Step 2: Literature Review
A comprehensive review of existing literature helps establish a solid foundation of knowledge and informs the subsequent steps in the research process.

<details>
<summary><b>1. SwinFace</b></summary><br/>
  
**Title**: SwinFace: A Multi-task Transformer for Face Recognition, Expression Recognition, Age Estimation and Attribute Estimation

The paper introduces SwinFace, a multi-task algorithm for face recognition, facial expression recognition, age estimation, and face attribute estimation. It utilizes a single Swin Transformer with task-specific subnets and a Multi-Level Channel Attention module to address conflicts and adaptively select optimal features. Experimental results demonstrate superior performance, achieving state-of-the-art accuracy of 90.97% on facial expression recognition (RAF-DB) and 0.22 error on age estimation (CLAP2015). The code and models are publicly available at [https://github.com/lxq1000/SwinFace]

<p align="center">
  <img src="https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/SwinFace_paper.png" alt="Image" />
</p>

</details>

<details>
<summary><b>2. Unraveling the Age Estimation Puzzle</b></summary><br/>
Comparative Analysis of Deep Learning Approaches for Facial Age Estimation
The paper addresses the challenge of comparing different age estimation methods due to inconsistencies in benchmarking processes. It challenges the notion that specialized methods are necessary for age estimation tasks and argues that the standard approach of utilizing cross-entropy loss is sufficient. Through a systematic analysis of various factors, including facial alignment, coverage, image resolution, representation, model architecture, and data amount, the paper finds that these factors often have a greater impact on age estimation results than the choice of the specific method. The study emphasizes the importance of consistent data preprocessing and standardized benchmarks for reliable and meaningful comparisons.
</details>

<details>
<summary><b>3. MiVOLO</b></summary><br/>
MiVOLO: Multi-input Transformer for Age and Gender Estimation  

This paper introduces MiVOLO, a method for age and gender estimation that utilizes a vision transformer and integrates both tasks into a unified dual input/output model. By incorporating person image data in addition to facial information, the model demonstrates improved generalization and the ability to estimate age and gender even when the face is occluded. Experimental results on multiple benchmarks show state-of-the-art performance and real-time processing capabilities. The model's age recognition performance surpasses human-level accuracy across various age ranges. The code, models, and additional dataset annotations are publicly available for validation and inference.  

<p align="center">
  <img src="https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/MiVOLO_paper.png" alt="Image" />
</p>

</details>

<details>
<summary><b>4. Rank consistent ordinal regression for neural networks with application to age estimation</b></summary><br/>
Rank consistent ordinal regression for neural networks with application to age estimation
The paper addresses the issue of capturing relative ordering information in class labels for tasks like age estimation. It introduces the COnsistent RAnk Logits (CORAL) framework, which transforms ordinal targets into binary classification subtasks to resolve inconsistencies among binary classifiers. The proposed method, applicable to various deep neural network architectures, demonstrates strong theoretical guarantees for rank-monotonicity and consistent confidence scores. Empirical evaluation on face-image datasets for age prediction shows a significant reduction in prediction error compared to reference ordinal regression networks.
</details>

<details>
<summary><b>5. Deep Regression Forests for Age Estimation</b></summary><br/>
Deep Regression Forests for Age Estimation
The paper introduces Deep Regression Forests (DRFs) as an end-to-end model for age estimation from facial images. DRFs address the challenge of heterogeneous facial feature space by jointly learning input-dependent data partitions and data abstractions. The proposed method achieves state-of-the-art results on three standard age estimation benchmarks, demonstrating its effectiveness in capturing the nonlinearity and variation in facial appearance across different ages.
</details>

<details>
<summary><b>6. Adaptive Mean-Residue Loss for Robust Facial Age Estimation</b></summary><br/>
Adaptive Mean-Residue Loss for Robust Facial Age Estimation
Automated facial age estimation has diverse real-world applications in multimedia analysis, e.g., video surveillance, and human-computer interaction. However, due to the randomness and ambiguity of the aging process, age assessment is challenging. Most research work over the topic regards the task as one of age regression, classification, and ranking problems, and cannot well leverage age distribution in representing labels with age ambiguity. In this work, we propose a simple yet effective loss function for robust facial age estimation via distribution learning, i.e., adaptive mean-residue loss, in which, the mean loss penalizes the difference between the estimated age distribution's mean and the ground-truth age, whereas the residue loss penalizes the entropy of age probability out of dynamic top-K in the distribution. Experimental results in the datasets FG-NET and CLAP2016 have validated the effectiveness of the proposed loss. Our code is available at [https://github.com/jacobzhaoziyuan/AMR-Loss]

<p align="center">
  <img src="https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/ADAPTIVE_MEAN_paper.png" alt="Image" />
</p>

</details>

<details>
<summary><b>7. FaceXFormer: A Unified Transformer for Facial Analysis</b></summary><br/>
It employs an encoder-decoder architecture, extracting multi-scale features from the input face image I, and fusing them into a unified representation F via MLP-Fusion. Task tokens T are processed alongside face representation F in the decoder, resulting in refined task-specific tokens T^. These refined tokens are then used for task-specific predictions by passing through the unified head.

<p align="center">
  <img src="https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/FaceXFormer_paper.png" alt="Image" />
</p>

</details>


**A Summary table for the above methods** 

| **Method**                                                                                | **Summary**                                                                                                                                                                        | **Code**                                                    | **Key Features**                          |
|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------|
| SwinFace                                                                                  | Swin Transformer and a Multi-Level  Channel Attention module to address  conflicts and select optimal features                                                                     | https://github.com/lxq1000/SwinFace                         | Transformers                              |
| Unraveling the Age Estimation Puzzle                                                      | A Benchmark for Age estimation                                                                                                                                                     | https://github.com/paplhjak/facial-age-estimation-benchmark | Benchmark                                 |
| MiVOLO                                                                                    | Estimate age and gender even when the face is occluded                                                                                                                             | https://github.com/WildChlamydia/MiVOLO                     | Age and Gender Estimation                 |
| Rank consistent ordinal regression for neural networks with application to age estimation | introduces the COnsistent RAnk Logits (CORAL) framework, which transforms ordinal targets into binary classification subtasks. | https://github.com/Raschka-research-group/coral-cnn         | Significant reduction in prediction error |
| Deep Regression Forests for Age Estimation                                                | Capturing the nonlinearity and variation in facial appearance across different ages                                                                                                | https://github.com/Sumching/Deep_Regression_Forests         |                                           |
| Adaptive Mean-Residue Loss for Robust Facial Age Estimation                               | Effective loss function for robust facial age estimation                                                                                                                           | https://github.com/jacobzhaoziyuan/AMR-Loss                 | Adaptive loss fn                          |
| FaceXFormer                                                |  A comprehensive range of facial analysis tasks such as face parsing, landmark detection, head pose estimation, attributes recognition, and estimation of age, gender, race, and landmarks visibility.               | https://github.com/Kartik-3004/facexformer | For almost all tasks in face

Note: I prepared this table with this amazing website: https://www.tablesgenerator.com/markdown_tables
</details>

# Step 3: Choose the appropriate method
The ResNet-50 model combined with regression is a powerful approach for facial age estimation. ResNet-50 is a deep convolutional neural network architecture that has proven to be highly effective in various computer vision tasks. By utilizing its depth and skip connections, ResNet-50 can effectively capture intricate facial features and patterns essential for age estimation. The regression component of the model enables it to directly predict the numerical age value, making it suitable for continuous age estimation rather than discrete age classification. This combination allows the model to learn complex relationships between facial attributes and age, providing accurate and precise age predictions. Overall, the ResNet-50 model with regression offers a robust and reliable solution for facial age estimation tasks.

## This is the diagram of proposed model  

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/method.png)  

# Step 4: Implementation of the selected method
## Dataset

In this sub-section we see some important datasets in the field of Facial age estimation:

<details>
<summary><b>1. MORPH</b></summary><br/>
MORPH is a facial age estimation dataset, which contains 55,134 facial images of 13,617 subjects ranging from 16 to 77 years old.
</details>

<details>
<summary><b>2. Adience</b></summary><br/>
The Adience dataset, published in 2014, contains 26,580 photos across 2,284 subjects with a binary gender label and one label from eight different age groups, partitioned into five splits. The key principle of the data set is to capture the images as close to real world conditions as possible, including all variations in appearance, pose, lighting condition and image quality, to name a few. For more information and download the dataset, please go to (https://talhassner.github.io/home/projects/Adience/Adience-data.html)
</details>

<details>
<summary><b>3. CACD (Cross-Age Celebrity Dataset)</b></summary><br/>
The Cross-Age Celebrity Dataset (CACD) contains 163,446 images from 2,000 celebrities collected from the Internet. The images are collected from search engines using celebrity name and year (2004-2013) as keywords. Therefore, it is possible to estimate the ages of the celebrities on the images by simply subtract the birth year from the year of which the photo was taken. For more information, please go to (https://bcsiriuschen.github.io/CARC/)       

</details>

<details>
<summary><b>4. FG-NET</b></summary><br/>
FGNet is a dataset for age estimation and face recognition across ages. It is composed of a total of 1,002 images of 82 people with age range from 0 to 69 and an age gap up to 45 years.
</details>

<details>
<summary><b>5. UTKFace</b></summary><br/>
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.
- consists of 20k+ face images in the wild (only single face in one image)
- provides the correspondingly aligned and cropped faces
- provides the corresponding landmarks (68 points)
- images are labelled by age, gender, and ethnicity

For more information and download, please refer to the [UTKFace](https://susanqq.github.io/UTKFace/)   
In this project, I downloaded cropped version of UTKFace from [Kaggle](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped?select=utkcropped)

</details>

**Note**: In this project, we use UTKFace. Please download from this link and use `utkcropped` folder to create csv: [Kaggle](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped?select=utkcropped)

### 1. EDA (Exploratory Data Analysis)

This repository contains code for performing exploratory data analysis on the UTK dataset, which consists of images categorized by age, gender, and ethnicity.

#### Contents

<details>
  <summary><b>I. Plot the Images in the UTK Dataset and create a csv file</b></summary><br/>

<details>
  <summary><b>1. Displaying sample images</b></summary><br/>

This may include loading and displaying sample images, obtaining image statistics, or performing basic image processing tasks.  

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/show_rand_samples.png) 
</details>

<details>
  <summary><b>2. Create a CSV File with Labels</b></summary><br/>
The labels may include information such as age, gender, and ethnicity for each image in the dataset.  

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/csv_file.png)  

</details>

</details>

<details>
<summary><b>II. Univariate Analysis</b></summary><br/>
Univariate analysis is a type of exploratory data analysis (EDA) that focuses on examining one variable at a time.
These histograms can provide insights into the dataset's composition and help identify any imbalances or patterns. 

<details>
  <summary><b>1. Histogram for Age</b></summary><br/>

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/age_histogram.png)  
</details>

<details>
  <summary><b>2. Histogram for Gender</b></summary><br/>

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/gender_histogram.png) 
</details>

<details>
  <summary><b>3. Histogram for Ethnicity</b></summary><br/>

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/ethnicity_histogram.png)
</details>

</details>


<details>
<summary><b>III. Bivariate Analysis</b></summary><br/>
Bivariate analysis examines relationships between two variables

<details>
<summary><b>1. Cross-tabulation of gender and ethnicity</b></summary><br/>
Calculating the cross-tabulation of gender and ethnicity using the `pandas.crosstab()` function. 
  This analysis can reveal the relationship between gender and ethnicity within the dataset and provide useful insights.  

```python
cross_tab = pd.crosstab(df['gender'], df['ethnicity'])`  
print(cross_tab)
```

   ![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/cross-tabulation.png)
</details>

<details>
<summary><b>2. Create Violin plots and Box Plots for Age (Separated by Gender)</b></summary><br/>
These plots can help identify any differences or patterns in the age distribution between men and women in the UTK dataset.  

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/violin_plot_age_men_women.png)
</details>


<details>
<summary><b>3. Create Violin Plots and Box Plots for Age (Separated by Ethnicity)</b></summary><br/>
These plots can help identify any differences or patterns in the age distribution among different ethnicities in the UTK dataset.  

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/violin_plot_Separated_by_Ethnicity.png)
</details>

</details>

### 2. Dataset Splitting

As you can saw in the **univariate analysis** section, the distribution of age isn't balance, and it's better to use **stratified sampling** to consider imbalancing in the age feature.

<details>
  <summary><b>1. Stratified sampling, and save csv</b></summary><br/>
  
The stratified sampling works by dividing the dataset into groups based on the values of the stratification feature (in this case, age). It then randomly samples from each group to create the train and test sets. The goal is to maintain the same proportion of different age groups in both sets, which helps ensure that the models trained on the training set generalize well to unseen data with different age distributions.

By using stratified sampling, you can obtain a representative train-test split that preserves the distribution of the age feature, which can be useful for building models that are robust across different age groups.

We can do stratify sampling with this code:

```python
df = pd.read_csv("./csv_dataset/utkface_dataset.csv")
df_train, df_temp = train_test_split(df, train_size=0.8, stratify=df.age, random_state=42)
df_test, df_valid = train_test_split(df_temp, train_size=0.5, stratify=df_temp.age, random_state=42) 
```

</details>

<details>
  <summary><b>2. Save and Plot the Training, Validation, and Test sets in separate CSV files</b></summary><br/>

Save the training, validation, and test sets in separate CSV files:   

```python
df_train.to_csv('./csv_dataset/train_set.csv', index=False) 
df_valid.to_csv('./csv_dataset/valid_set.csv', index=False)  
df_test.to_csv('./csv_dataset/test_set.csv', index=False)
```

And now, plot each histograms with this lines of code:  

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(df_train.age, bins=len(df_train.age.unique())); axes[0].set_title('Train') 
axes[1].hist(df_valid.age, bins=len(df_valid.age.unique())); axes[1].set_title('Validation')
axes[2].hist(df_test.age, bins=len(df_test.age.unique())); axes[2].set_title('Test')
```

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/histogram_train_valid_test.png)

This histograms will help ensure that the distributions of age in these sets are similar, indicating a balanced and representative dataset split.  
This step is crucial for further analysis or modeling tasks, as it allows you to access and manipulate each set individually.
</details>

### 3. Transformations

The defined transformations include resizing images, applying random flips and rotations, adjusting image color, converting images to tensors, and normalizing pixel values.
I wrote all of them in `Custom-dataset_dataloader.py` file.

### 4. Custom Dataset and DataLoader

The custom dataset allows you to load and preprocess your own data, while the dataloader provides an efficient way to iterate over the dataset during training or evaluation.

#### Contents

1. [Custom Dataset](#custom-dataset)
2. [Define DataLoader](#define-dataloader)

<details>
  <summary><b>1. Custom Dataset</b></summary><br/>
The custom dataset is designed to handle your specific data format and apply any necessary preprocessing steps. You can modify the dataset class according to your data structure, file paths, and preprocessing requirements.
</details>

<details>
  <summary><b>2. DataLoader</b></summary><br/>
The dataloader is responsible for efficiently loading and batching the data from the custom dataset. It provides an iterator interface that allows you to easily access the data during model training or evaluation. You can customize the dataloader settings such as batch size, shuffling, and parallel data loading based on your specific needs.
</details>

## 5. Model

The models used in this project are ResNet50 and EfficientNet B0.

### Contents

1. [ResNet50 Model](#resnet50-model)
2. [EfficientNet B0 Model](#efficientnet-b0-model)
3. [Vision transformer](#efficientnet-b0-model)

<details>
  <summary><b>1. ResNet50</b></summary><br/>
The ResNet50 architecture is a widely-used convolutional neural network that has shown impressive performance on various computer vision tasks. You will learn how to load the pre-trained ResNet50 model, fine-tune it on your custom dataset, and use it for inference.  

   ![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/Resnet50.png)
   
Define Resnet:    

```python
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to(device)
```

</details>

<details>
  <summary><b>2. EfficientNet B0</b></summary><br/>
EfficientNet is a family of convolutional neural networks that have achieved state-of-the-art performance on image classification tasks while being computationally efficient. You will learn how to load the pre-trained EfficientNet B0 model, adapt it to your custom dataset, and leverage its capabilities for classification or feature extraction.  

   ![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/EfficientNet.png)
   
Define Efficientnet:   

```python
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='efficientnet', pretrain_weights='IMAGENET1K_V1').to(device)
```

</details>

<details>
  <summary><b>3. Vision Transformer</b></summary><br/>
A vision transformer (ViT) is a transformer designed for computer vision.[1] A ViT breaks down an input image into a series of patches (rather than breaking up text into tokens), serialises each patch into a vector, and maps it to a smaller dimension with a single matrix multiplication. These vector embeddings are then processed by a transformer encoder as if they were token embeddings.

   ![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/Vision_Transformer.gif) 
   
Define Vision Transformer:  

```python
model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='vit', pretrain_weights=True).to(device)
```

</details>


## 6. Training Process

This repository contains code for the training process of a model, including finding hyperparameters, the training and evaluation loop, and plotting learning curves.

### Contents

1. [Finding Hyperparameters](#finding-hyperparameters)
   1. [Step 1: Calculate the Loss for an Untrained Model](#step-1-calculate-the-loss-for-an-untrained-model-using-a-few-batches)
   2. [Step 2: Train and Overfit the Model on a Small Subset of the Dataset](#step-2-try-to-train-and-overfit-the-model-on-a-small-subset-of-the-dataset)
   3. [Step 3: Train the Model for a Limited Number of Epochs](#step-3-train-the-model-for-a-limited-number-of-epochs-experimenting-with-various-learning-rates)
   4. [Step 4: Create a Small Grid Using Weight Decay and the Best Learning Rate and save it to a CSV file](#step-4-create-a-small-grid-using-the-weight-decay-and-the-best-learning-rate-and-save-it-to-a-CSV-file)
   5. [Step 5: Train the Model for Longer Epochs Using the Best Model from Step 4](#step-5-train-model-for-longer-epochs-using-the-best-model-from-step-4)
2. [Training and Evaluation Loop](#train-and-evaluation-loop)
3. [Plotting Learning Curves with Matplotlib and TensorBoard](#plot-learning-curves)
4. [Save the best model from .pt to .jit](#Save-the-best-model-from-.pt-to-.jit)

#### Finding Hyperparameters

The process involves several steps, including calculating the loss for an untrained model, overfitting the model on a small subset of the dataset, training the model for a limited number of epochs with various learning rates, creating a small grid using weight decay and the best learning rate, and finally training the model for longer epochs using the best model from the previous step.

<details>
  <summary><b>Step 1: Calculate the loss for an untrained model using one batch</b></summary><br/>
This step helps us to understand that the forward pass of the model is working. The forward pass of a neural network model refers to the process of propagating input data through the model's layers to obtain predictions or output values.

This is code for step 1 in `hyperparameters_tuning.py`:

```python
x_batch, y_batch, _, _ = next(iter(train_loader)) 
outputs = model(x_batch.to(device))
loss = loss_fn(outputs, y_batch.to(device)) 
print(loss) 
```
</details>

<details>
  <summary><b>Step 2: Train and overfit the model on a small subset of the dataset</b></summary><br/>
The goal of Step 2 is to train the model on a small subset of the dataset to assess its ability to learn and memorize the training data.
  
```python
_, mini_train_dataset = random_split(train_set, (len(train_set)-1000, 1000)) 
mini_train_loader = DataLoader(mini_train_dataset, 5) 

num_epochs = 5
for epoch in range(num_epochs):  
    model, loss_train, train_metric = train_one_epoch(model, mini_train_loader, loss_fn, optimizer, metric, epoch=epoch) 
```
 
</details>

<details>
  <summary><b>Step 3: Train the model for a limited number of epochs, experimenting with various learning rates</b></summary><br/>
This step helps us to identify the learning rate that leads to optimal training progress and convergence.  

```python
for lr in [0.001, 0.0001, 0.0005]:
    print(f'lr is: {lr}')
    model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='efficientnet',pretrain_weights='IMAGENET1K_V1').to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_epochs):
        model, loss_train, train_metric = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=epoch)
    print('')
```
</details>

<details>
  <summary><b>Step 4: Create a small grid using weight decay and the best learning rate and save it to a CSV file</b></summary><br/>
The goal of Step 4 is to create a small grid using weight decay and the best learning rate, and save it to a CSV file. This grid allows us to examine how weight decay regularization impacts the performance of the model.

```python
small_grid_list = []
for lr in [0.0005, 0.0008, 0.001]: 
    for wd in [1e-4, 1e-5, 0.]: 
        print(f'LR={lr}, WD={wd}')
        model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='efficientnet', pretrain_weights='IMAGENET1K_V1').to(device)
        loss_fn = nn.L1Loss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        for epoch in range(num_epochs):
            model, loss_train, train_metric = train_one_epoch(model, mini_train_loader, loss_fn, optimizer, metric, epoch=epoch)
        small_grid_list.append([lr, wd, loss_train])
```

</details>

<details>
  <summary><b>Step 5: Train the model for longer epochs using the best model from step 4</b></summary><br/>
The goal of Step 5 is to train the model for longer epochs using the best model obtained from Step 4. This step aims to maximize the model's learning potential and achieve improved performance by allowing it to learn from the data for an extended period.  
  
Please refer to `train.py`
</details>

<details>
  <summary><b>Step 6: Save the best model from .pt to .jit</b></summary><br/>
The goal of this step is to convert the best model from .pt to .jit format. This conversion is primarily done to optimize and enhance the model's performance during deployment.
</details>

#### Train and Evaluation Loop

The train loop handles the training process, including forward and backward passes, updating model parameters, and monitoring training metrics. The evaluation loop performs model evaluation on a separate validation or test dataset and computes relevant evaluation metrics.

<details>
  <summary><b>Plotting Learning Curves with Matplotlib and TensorBoard</b></summary><br/>
Learning curves visualize the model's training and validation performance over epochs, providing insights into the model's learning progress, convergence, and potential issues such as overfitting or underfitting.\
TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.  

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/loss-tensorboard.png)  
</details>

#### Inference function
Define the inference function (`inference.py`): The inference function takes a pre-trained Age Estimation model, an input image path, and an output image path. It loads the model checkpoint, performs inference on the input image, and saves the output image with the estimated age.

**Run the inference**: Call the inference function with the loaded model, input image path, and output image path. 
The function will process the image, estimate the age, and save the output image with the estimated age written on it.

<details>
  <summary><b>Inference pipeline</b></summary><br/>

![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/age_estimation_inference.png)  

</details>

## Todo

...

### Contents

#### Inference

- [✔️] Implement code for performing inference using the trained model.
- [✔️] Provide instructions on how to use the inference code with sample input data.

#### Experiments

##### Train and Evaluate the Model Using Various Datasets

- [ ] Conduct experiments to train and evaluate the model using different datasets.
- [ ] Document the datasets used, training process, and evaluation results.
- [ ] Provide guidelines on how to adapt the code for using custom datasets.

##### Train the Model Using One Dataset and Test it on a Different One

- [ ] Perform experiments to train the model on one dataset and evaluate its performance on a different dataset.
- [ ] Describe the process of training and testing on different datasets.
- [ ] Report the evaluation metrics and discuss the results.

##### Analyze the Loss Value with Respect to Age, Gender, and Race

- [ ] Analyze the loss value of the model with respect to age, gender, and race.
- [ ] Provide code or scripts to calculate and visualize the loss values for different demographic groups.
- [ ] Discuss the insights and implications of the analysis.

##### Analyze the Model's Sensitivity

- [ ] Conduct sensitivity analysis to understand the model's response to variations in input data.
- [ ] Outline the methodology and metrics used for sensitivity analysis.
- [ ] Present the findings and interpretations of the sensitivity analysis.

##### Create a Heatmap for the Face Images

- [ ] Develop code to generate heatmaps for face images based on the model's predictions or activations.
- [ ] Explain the process of creating heatmaps and their significance in understanding the model's behavior.
- [ ] Provide examples and visualizations of the generated heatmaps.

##### Use the Model to Perform Age Estimation on a Webcam Image

- [ ] Integrate the model with webcam functionality to perform age estimation on real-time images.
- [ ] Detail the steps and code required to use the model for age estimation on webcam images.
- [ ] Include any necessary dependencies or setup instructions.
