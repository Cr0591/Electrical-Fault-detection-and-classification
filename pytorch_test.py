import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

df = pd.read_csv(r'classData.csv', usecols=[
    'G', 'C', 'B', 'A', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])

l = np.array(df).tolist()
classData = torch.DoubleTensor(l)
print(classData)

data.TensorDataset
