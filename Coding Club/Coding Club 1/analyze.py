from tkinter import Tk
from tkinter.filedialog import askopenfile
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Read file
Tk().withdraw()
file = askopenfile()
filename = file.name

# Open CSV
dataframe = pd.read_csv(filename)

# Cleaning
cleaned_df = dataframe.dropna()
cleaned_df = cleaned_df.drop_duplicates()
cleaned_df = cleaned_df.drop("id", axis=1)

dmap = LabelEncoder()

# Split DF into subsets
cleaned_df['diagnosis'] = dmap.fit_transform(cleaned_df['diagnosis'])

# Show plot

corr_matrix = cleaned_df.corr()

plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True, square=True)
plt.title('Correlation Matrix')
plt.xlabel('Feature Index')
plt.ylabel("Feature Value")
plt.show()