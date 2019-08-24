import pandas as pd
from scipy.io import wavfile
import numpy as np

df = pd.read_csv('./instruments.csv')


# Add length to all items
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('./wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate


#
classes = np.unique(df.label)
classes_dist = df.groupby(['label']).length.mean()


print(classes_dist)