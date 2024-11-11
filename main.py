import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
#start from here

readable  = pd.read_csv('teleCust1000t.csv')
print(readable.head())