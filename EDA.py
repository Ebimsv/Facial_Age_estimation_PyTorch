import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv('csv_dataset/utkface_dataset.csv')
print(df.head())

# Plot histogram for `age, ethnicity, and gender`
def plot_histograms_with_names(dataframe):
    # Get the column names from the dataframe
    columns = dataframe.columns

    # Create subplots for each column
    fig, axes = plt.subplots(nrows=len(columns)-1, ncols=1, figsize=(8, 6 * len(columns)))

    # Plot histogram for each column and write column name above each subplot
    for i, column in enumerate(columns[1:]):
        ax = axes[i]
        sns.histplot(data=dataframe[column], ax=ax)
        ax.set_xlabel(column)
        ax.set_ylabel('Counts')
        ax.set_title(f'Histogram of {column}')
        
    plt.tight_layout()
    plt.show()

# Example usage: 
plot_histograms_with_names(df)

# 1. Calculate the cross-tabulation of gender and ethnicity using the pandas.crosstab() function.
cross_tab = pd.crosstab(df['gender'], df['ethnicity'])
print(cross_tab)

# 2. Create violin plots and box plots for age, separately for men and women.
men_df = df[df['gender'] == 'Male']
women_df = df[df['gender'] == 'Female']

plt.figure(figsize=(12, 8))

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

# 3. Create violin plots and box plots for age, separately for each ethnicity.
Asian_df = df[df['ethnicity'] == 'Asian']
Black_df = df[df['ethnicity'] == 'Black']
Indian_df = df[df['ethnicity'] == 'Indian']
Others_df = df[df['ethnicity'] == 'Others']
White_df = df[df['ethnicity'] == 'White']

plt.figure(figsize=(12, 8))

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
