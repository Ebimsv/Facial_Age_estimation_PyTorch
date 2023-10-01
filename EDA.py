import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('csv_dataset/utkface_dataset.csv')
print(df.head())

# 1. Plot histogram for age
print(df.hist())

# 2. Plot histogram for gender
category_counts = df['gender'].value_counts()
plt.bar(range(len(category_counts)), category_counts.values)
plt.ylabel('Counts')
plt.title('Histogram for gender')
plt.xticks(range(len(category_counts)), category_counts.index, rotation=90)
plt.show()

# 3. Plot histogram for ethnicity
category_counts = df['ethnicity'].value_counts()
plt.bar(range(len(category_counts)), category_counts.values)
plt.ylabel('Counts')
plt.title('Histogram for ethnicity')
plt.xticks(range(len(category_counts)), category_counts.index, rotation=90)
plt.show()

# 4. Calculate the cross-tabulation of gender and ethnicity using the pandas.crosstab() function.
cross_tab = pd.crosstab(df['gender'], df['ethnicity'])
print(cross_tab)

# 5. Create violin plots and box plots for age, separately for men and women.
men_df = df[df['gender'] == 'Male']
women_df = df[df['gender'] == 'Female']

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.violinplot(men_df['age'], vert=False)
plt.boxplot(men_df['age'], vert=False)
plt.xlabel('Age')
plt.title('Violin Plot - Men')

plt.subplot(1, 2, 2)
plt.violinplot(women_df['age'], vert=False)
plt.boxplot(women_df['age'], vert=False)
plt.xlabel('Age')
plt.title('Violin Plot - Women')

# 6. Create violin plots and box plots for age, separately for each ethnicity.
Asian_df = df[df['ethnicity'] == 'Asian']
Black_df = df[df['ethnicity'] == 'Black']
Indian_df = df[df['ethnicity'] == 'Indian']
Others_df = df[df['ethnicity'] == 'Others']
White_df = df[df['ethnicity'] == 'White']

plt.figure(figsize=(10, 15))

plt.subplot(3, 2, 1)
plt.violinplot(Asian_df['age'], vert=False)
plt.boxplot(women_df['age'], vert=False)
plt.xlabel('Age', color='b')
plt.title('Asian ethnicity plot', color='r')

plt.subplot(3, 2, 2)
plt.violinplot(Black_df['age'], vert=False)
plt.boxplot(Black_df['age'], vert=False)
plt.xlabel('Age', color='b')
plt.title('Black ethnicity plot', color='r')

plt.subplot(3, 2, 3)
plt.violinplot(Indian_df['age'], vert=False)
plt.boxplot(Indian_df['age'], vert=False)
plt.xlabel('Age', color='b')
plt.title('Indian ethnicity plot', color='r')

plt.subplot(3, 2, 4)
plt.boxplot(Others_df['age'], vert=False)
plt.boxplot(Others_df['age'], vert=False)
plt.xlabel('Age', color='b')
plt.title('Others ethnicity plot', color='r')

plt.subplot(3, 2, 5)
plt.boxplot(White_df['age'], vert=False)
plt.boxplot(White_df['age'], vert=False)
plt.xlabel('Age', color='b')
plt.title('White ethnicity plot', color='r')

# 6. Plot histograms for age in the training, validation, and test sets.
df_train, df_test = train_test_split(df, train_size=0.8, random_state=42)
df_train, df_valid = train_test_split(df_train, train_size=0.85, random_state=42)

# Warning: Ensure that the distributions of the training, validation, and test sets are similar.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(df_train.age, bins=len(df_train.age.unique())); axes[0].set_title('Train')
axes[1].hist(df_valid.age, bins=len(df_valid.age.unique())); axes[1].set_title('Validation')
axes[2].hist(df_test.age, bins=len(df_test.age.unique())); axes[2].set_title('Test')
