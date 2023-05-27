
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from IPython.display import display
import tensorflow as tf

data = pd.read_csv('ifood_df.csv', delimiter = ';', header='infer')
display(data.head())
data.describe()
