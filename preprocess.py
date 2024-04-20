import numpy as np
import pandas as pd

seed = 2024

arr = np.arange(228)
np.random.seed(seed)
np.random.shuffle(arr)

train = arr[:int(0.8*len(arr))]
valid = arr[int(0.8*len(arr)):]
test = np.arange(228, 252)

train = pd.DataFrame(train)
valid = pd.DataFrame(valid)
test = pd.DataFrame(test)

train.to_csv('data/train.csv', index=False)
valid.to_csv('data/valid.csv', index=False)
test.to_csv('data/test.csv', index=False)