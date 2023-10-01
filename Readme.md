![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/Age-Estimation.png)

# About This Project
This repository provides a detailed, step-by-step implementation guide for facial age estimation using PyTorch, a popular deep learning framework. The goal of this project is to accurately estimate the age of individuals based on their facial appearance.

# Step 1: Accurate and concise definition of the problem
Age estimation refers to the process of estimating a person's age based on various observable characteristics or features. It is often performed using computer vision techniques that analyze facial attributes such as wrinkles, skin texture, and hair color. By comparing these characteristics with a database of known age examples, algorithms can make an educated guess about a person's age. However, it's important to note that age estimation is not always accurate and can be influenced by factors such as lighting conditions, facial expressions, and individual variations. It is primarily used in applications like biometrics, demographic analysis, or age-restricted access control systems.

## The goal of solving a problem or challenge
The goals of solving a problem or challenge in facial age estimation typically revolve around improving the accuracy and reliability of estimating a person's age based on their facial appearance. Here are some specific goals that researchers and developers in this field aim to achieve:

1. **Improved Accuracy**: One of the primary goals is to enhance the accuracy of facial age estimation algorithms. This involves reducing the margin of error in estimating a person's age and increasing the precision of the predictions. The ultimate aim is to develop models that can estimate age with a high level of accuracy, approaching or surpassing human-level performance.

2. **Reduced Bias**: Another important goal is to minimize biases in facial age estimation algorithms. Biases can arise due to factors such as ethnicity, gender, or other demographic characteristics. It is crucial to develop models that are fair and unbiased, providing accurate age estimates regardless of an individual's background or appearance.

3. **Robustness**: Facial age estimation algorithms should be robust and reliable under various conditions and scenarios. They should be able to handle variations in lighting, pose, expression, and other factors that may affect facial appearance. Robustness ensures that age estimation models perform consistently and accurately across different environments and datasets.

4. **Generalization**: Age estimation algorithms should have good generalization capabilities. They should be trained on diverse datasets representing different populations, age groups, and ethnicities. The goal is to develop models that can accurately estimate age for individuals from various demographics and cultural backgrounds, rather than being limited to a specific subset of the population.

5. **Real-Time Performance**: In many applications, such as surveillance systems, age estimation algorithms need to operate in real-time. Therefore, a goal is to develop efficient algorithms that can provide age estimates quickly, allowing for practical implementation in real-world scenarios without significant computational overhead.

6. **Cross-Domain Adaptability**: Facial age estimation algorithms should be adaptable to different domains and applications. For example, they should be effective in estimating age from photographs, video frames, or even in live video streams. The ability to adapt to different data sources and domains expands the potential applications of age estimation technology.

7. **Privacy and Ethical Considerations**: It is important to address privacy concerns and ethical considerations associated with facial age estimation. The development of age estimation algorithms should take into account data protection, consent, and potential misuse. Ensuring that privacy is respected and ethical guidelines are followed is a crucial goal in this field.

By striving to achieve these goals, researchers and developers aim to advance the field of facial age estimation and create algorithms that can have practical applications in areas such as biometrics, demographics analysis, age-specific marketing, and personalized user experiences.

# Step 2: Literature Review
## 1. SwinFace: A Multi-task Transformer for Face Recognition, Expression Recognition, Age Estimation and Attribute Estimation
The paper introduces SwinFace, a multi-task algorithm for face recognition, facial expression recognition, age estimation, and face attribute estimation. It utilizes a single Swin Transformer with task-specific subnets and a Multi-Level Channel Attention module to address conflicts and adaptively select optimal features. Experimental results demonstrate superior performance, achieving state-of-the-art accuracy of 90.97% on facial expression recognition (RAF-DB) and 0.22 error on age estimation (CLAP2015). The code and models are publicly available at [https://github.com/lxq1000/SwinFace. ↗](https://github.com/lxq1000/SwinFace.)\

## 2. Unraveling the Age Estimation Puzzle: Comparative Analysis of Deep Learning Approaches for Facial Age Estimation
The paper addresses the challenge of comparing different age estimation methods due to inconsistencies in benchmarking processes. It challenges the notion that specialized methods are necessary for age estimation tasks and argues that the standard approach of utilizing cross-entropy loss is sufficient. Through a systematic analysis of various factors, including facial alignment, coverage, image resolution, representation, model architecture, and data amount, the paper finds that these factors often have a greater impact on age estimation results than the choice of the specific method. The study emphasizes the importance of consistent data preprocessing and standardized benchmarks for reliable and meaningful comparisons. The source code is available at Facial-Age-Benchmark.

## 3. MiVOLO: Multi-input Transformer for Age and Gender Estimation
The paper introduces MiVOLO, a method for age and gender estimation that utilizes a vision transformer and integrates both tasks into a unified dual input/output model. By incorporating person image data in addition to facial information, the model demonstrates improved generalization and the ability to estimate age and gender even when the face is occluded. Experimental results on multiple benchmarks show state-of-the-art performance and real-time processing capabilities. The model's age recognition performance surpasses human-level accuracy across various age ranges. The code, models, and additional dataset annotations are publicly available for validation and inference.

## 4. Rank consistent ordinal regression for neural networks with application to age estimation
The paper addresses the issue of capturing relative ordering information in class labels for tasks like age estimation. It introduces the COnsistent RAnk Logits (CORAL) framework, which transforms ordinal targets into binary classification subtasks to resolve inconsistencies among binary classifiers. The proposed method, applicable to various deep neural network architectures, demonstrates strong theoretical guarantees for rank-monotonicity and consistent confidence scores. Empirical evaluation on face-image datasets for age prediction shows a significant reduction in prediction error compared to reference ordinal regression networks.

## 5. Deep Regression Forests for Age Estimation
The paper introduces Deep Regression Forests (DRFs) as an end-to-end model for age estimation from facial images. DRFs address the challenge of heterogeneous facial feature space by jointly learning input-dependent data partitions and data abstractions. The proposed method achieves state-of-the-art results on three standard age estimation benchmarks, demonstrating its effectiveness in capturing the nonlinearity and variation in facial appearance across different ages.

## 6. Adaptive Mean-Residue Loss for Robust Facial Age Estimation
Automated facial age estimation has diverse real-world applications in multimedia analysis, e.g., video surveillance, and human-computer interaction. However, due to the randomness and ambiguity of the aging process, age assessment is challenging. Most research work over the topic regards the task as one of age regression, classification, and ranking problems, and cannot well leverage age distribution in representing labels with age ambiguity. In this work, we propose a simple yet effective loss function for robust facial age estimation via distribution learning, i.e., adaptive mean-residue loss, in which, the mean loss penalizes the difference between the estimated age distribution's mean and the ground-truth age, whereas the residue loss penalizes the entropy of age probability out of dynamic top-K in the distribution. Experimental results in the datasets FG-NET and CLAP2016 have validated the effectiveness of the proposed loss. Our code is available at https://github.com/jacobzhaoziyuan/AMR-Loss.


# Step 3: Choose the appropriate method
The ResNet-50 model combined with regression is a powerful approach for facial age estimation. ResNet-50 is a deep convolutional neural network architecture that has proven to be highly effective in various computer vision tasks. By utilizing its depth and skip connections, ResNet-50 can effectively capture intricate facial features and patterns essential for age estimation. The regression component of the model enables it to directly predict the numerical age value, making it suitable for continuous age estimation rather than discrete age classification. This combination allows the model to learn complex relationships between facial attributes and age, providing accurate and precise age predictions. Overall, the ResNet-50 model with regression offers a robust and reliable solution for facial age estimation tasks.
## This is the diagram of proposed model
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/method.png)  

# Step 4: Implementation of the selected method
## Dataset
### 1. EDA (Exploratory Data Analysis)

This repository contains code for performing exploratory data analysis on the UTK dataset, which consists of images categorized by age, gender, and ethnicity.

#### Contents

1. [Explore the Images in the UTK Dataset](#explore-the-images-in-the-utk-dataset)
2. [Create a CSV File with Labels](#create-a-csv-file-with-labels)
3. [Plot Histograms for Age, Gender, and Ethnicity](#plot-histograms-for-age-gender-and-ethnicity)
4. [Calculate Cross-Tabulation of Gender and Ethnicity](#calculate-cross-tabulation-of-gender-and-ethnicity)
5. [Create Violin Plots and Box Plots for Age (Separated by Gender)](#create-violin-plots-and-box-plots-for-age-separated-by-gender)
6. [Create Violin Plots and Box Plots for Age (Separated by Ethnicity)](#create-violin-plots-and-box-plots-for-age-separated-by-ethnicity)

##### Explore the Images in the UTK Dataset

- This may include loading and displaying sample images, obtaining image statistics, or performing basic image processing tasks.\
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/show_rand_samples.png) 

##### Create a CSV File with Labels

- The labels may include information such as age, gender, and ethnicity for each image in the dataset.\
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/csv_file.png)  

##### Plot Histograms for Age, Gender, and Ethnicity

- These histograms can provide insights into the dataset's composition and help identify any imbalances or patterns. \

   - Histogram for Age:
      ![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/age_histogram.png) \

   - Histogram for Gender:
      ![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/gender_histogram.png) \

   - Histogram for Ethnicity:
      ![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/ethnicity_histogram.png)


##### Calculate Cross-Tabulation of Gender and Ethnicity

- Calculating the cross-tabulation of gender and ethnicity using the `pandas.crosstab()` function. This analysis can reveal the relationship between gender and ethnicity within the dataset and provide useful insights. \ 
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/cross-tabulation.png)

##### Create Violin Plots and Box Plots for Age (Separated by Gender)

- These plots can help identify any differences or patterns in the age distribution between men and women in the UTK dataset.\
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/violin_plot_age_men_women.png)

##### Create Violin Plots and Box Plots for Age (Separated by Ethnicity)

- These plots can help identify any differences or patterns in the age distribution among different ethnicities in the UTK dataset.\
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/violin_plot_Separated_by_Ethnicity.png)

### 2. Dataset Splitting

This repository contains code for splitting datasets and analyzing the distributions of age in the training, validation, and test sets. Additionally, it provides instructions for saving these sets in separate CSV files.

#### Contents

1. [Plot Histograms for Age in the Training, Validation, and Test Sets](#plot-histograms-for-age-in-the-training-validation-and-test-sets)
2. [Save the Training, Validation, and Test Sets in Separate CSV Files](#save-the-training-validation-and-test-sets-in-separate-csv-files)

##### Plot Histograms for Age in the Training, Validation, and Test Sets

You will find code and instructions for plotting histograms to visualize the distribution of age in the training, validation, and test sets. The histograms will help ensure that the distributions of age in these sets are similar, indicating a balanced and representative dataset split.

##### Save the Training, Validation, and Test Sets in Separate CSV Files

This step is crucial for further analysis or modeling tasks, as it allows you to access and manipulate each set individually.


### 3. Transformations

The defined transformations include resizing images, applying random flips and rotations, adjusting image color, converting images to tensors, and normalizing pixel values.

#### Contents

1. [Resizing Images](#resizing-images)
2. [Applying Random Horizontal Flips](#applying-random-horizontal-flips)
3. [Introducing Random Rotations](#introducing-random-rotations)
4. [Adjusting Image Color using ColorJitter](#adjusting-image-color-using-colorjitter)
5. [Converting Images to Tensors](#converting-images-to-tensors)
6. [Normalizing Pixel Values](#normalizing-pixel-values)

##### Resizing Images

- Resizing images to a resolution of 128x128 pixels. Resizing the images ensures consistent dimensions and prepares them for further processing or analysis.

##### Applying Random Horizontal Flips

- Random flips can introduce diversity and prevent model bias towards specific orientations.

##### Introducing Random Rotations

- Random rotations can simulate variation and improve model robustness to different orientations.

##### Adjusting Image Color using ColorJitter

- ColorJitter allows you to modify the brightness, contrast, saturation, and hue of the images, enhancing their visual appearance and potentially improving model performance.

##### Converting Images to Tensors

- Converting images to tensors is a required step for many deep learning frameworks and enables efficient computation on GPUs.

##### Normalizing Pixel Values

- Normalizing the pixel values ensures that they have a standard range and distribution, making the training process more stable. The provided mean and standard deviation values (**mean**=[0.485, 0.456, 0.406], **std**=[0.229, 0.224, 0.225]) can be used for this normalization.

### 4. Custom Dataset and DataLoader

The custom dataset allows you to load and preprocess your own data, while the dataloader provides an efficient way to iterate over the dataset during training or evaluation.

#### Contents

1. [Custom Dataset](#custom-dataset)
2. [Define DataLoader](#define-dataloader)

##### Custom Dataset

- The custom dataset is designed to handle your specific data format and apply any necessary preprocessing steps. You can modify the dataset class according to your data structure, file paths, and preprocessing requirements.

##### Define DataLoader

- The dataloader is responsible for efficiently loading and batching the data from the custom dataset. It provides an iterator interface that allows you to easily access the data during model training or evaluation. You can customize the dataloader settings such as batch size, shuffling, and parallel data loading based on your specific needs.

## Model with Custom Dataset

This repository contains code for training and using models with a custom dataset. The models used in this project are ResNet50 and EfficientNet B0, and they are trained on the custom dataset you provide.

### Contents

1. [ResNet50 Model](#resnet50-model)
2. [EfficientNet B0 Model](#efficientnet-b0-model)

#### ResNet50 Model

- The ResNet50 architecture is a widely-used convolutional neural network that has shown impressive performance on various computer vision tasks. You will learn how to load the pre-trained ResNet50 model, fine-tune it on your custom dataset, and use it for inference. \
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/Resnet50.png)

#### EfficientNet B0 Model

- EfficientNet is a family of convolutional neural networks that have achieved state-of-the-art performance on image classification tasks while being computationally efficient. You will learn how to load the pre-trained EfficientNet B0 model, adapt it to your custom dataset, and leverage its capabilities for classification or feature extraction.
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/EfficientNet.png)

## Training Process

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

You will find code and instructions for finding the optimal hyperparameters for your model. The process involves several steps, including calculating the loss for an untrained model, overfitting the model on a small subset of the dataset, training the model for a limited number of epochs with various learning rates, creating a small grid using weight decay and the best learning rate, and finally training the model for longer epochs using the best model from the previous step.

##### Step 1: Calculate the Loss for an Untrained Model Using a Few Batches

- You will learn how to calculate the loss for an untrained model using a few batches of your dataset. This step helps you understand the initial performance of the model before any training.

##### Step 2: Train and Overfit the Model on a Small Subset of the Dataset

- You will learn how to train and overfit the model on a small subset of your dataset. Overfitting the model on a small subset helps you gauge the model's capacity to learn and memorize the training data.

##### Step 3: Train the Model for a Limited Number of Epochs, Experimenting with Various Learning Rates

- In this step, you will learn how to train the model for a limited number of epochs while experimenting with various learning rates. This step helps you identify the learning rate that leads to optimal training progress and convergence.

##### Step 4: Create a Small Grid Using Weight Decay and the Best Learning Rate and save it to a CSV file

- You will learn how to create a small grid using weight decay and the best learning rate obtained from the previous step. This grid helps you explore the effect of weight decay regularization on the model's performance.

##### Step 5: Train the Model for Longer Epochs Using the Best Model from Step 4

- You will learn how to train the model for longer epochs using the best model obtained from the previous step. This step allows you to maximize the model's learning potential and achieve improved performance.

##### Step 6: Save the best model from .pt to .jit
- The primary reason for this conversion is to optimize and improve the model's performance during deployment.

#### Train and Evaluation Loop

The train loop handles the training process, including forward and backward passes, updating model parameters, and monitoring training metrics. The evaluation loop performs model evaluation on a separate validation or test dataset and computes relevant evaluation metrics.

#### Plotting Learning Curves with Matplotlib and TensorBoard

Learning curves visualize the model's training and validation performance over epochs, providing insights into the model's learning progress, convergence, and potential issues such as overfitting or underfitting.\
TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.
![alt text](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/blob/main/pics/loss-tensorboard.png)  

## Todo

This repository contains code for various tasks related to inference and experiments with the model.

### Contents

1. [Inference](#inference)
2. [Experiments](#experiments)
   1. [Train and Evaluate the Model Using Various Datasets](#train-and-evaluate-the-model-using-various-datasets)
   2. [Train the Model Using One Dataset and Test it on a Different One](#train-the-model-using-one-dataset-and-then-test-it-on-a-different-one)
   3. [Analyze the Loss Value with Respect to Age, Gender, and Race](#analyze-the-loss-value-with-respect-to-age-gender-and-race)
   4. [Analyze the Model's Sensitivity](#analyze-the-models-sensitivity)
   5. [Create a Heatmap for the Face Images](#create-a-heatmap-for-the-face-images)
3. [Use the Model to Perform Age Estimation on a Webcam Image](#use-the-model-to-perform-age-estimation-on-a-webcam-image)

#### Inference

- [ ] Implement code for performing inference using the trained model.
- [ ] Provide instructions on how to use the inference code with sample input data.

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