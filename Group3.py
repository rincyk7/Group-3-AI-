import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import re



plt.style.use('seaborn-dark')

#Loading the CSV file
posts = pd.read_csv('/Users/AI/Group_project/Youtube05-Shakira.csv')

#Initial exploration of the dataframe
pd.set_option('display.max_columns', None)

posts.head()

posts.tail()

posts.info()

# Using a heatmaps to better visualize the integrity of the dataframe.
# The information provided by posts.info() is visualy confirmed. There are no missing data in this dataframe.
plt.figure(figsize = (16,8))
sns.heatmap(posts.isnull(), cmap = 'viridis', cbar = False)

#Columns 'COMMENT_ID nd DATE are droped. Both seem not to carry any pattern that will influence in the prediction of the target variable.
posts.drop(['COMMENT_ID', 'DATE'], axis=1, inplace = True)

posts.head()

# Checking for imbalanced target variable.
posts.CLASS.value_counts()

target_freq = pd.Series(posts.CLASS).value_counts(normalize=True)

print(target_freq)

# Plot the relative frequency of the binary target variable
plt.figure(figsize = (10,5))
sns.barplot(x = target_freq.index, y = target_freq)
plt.ylabel('Relative Frequency')
plt.xlabel('Classes')
