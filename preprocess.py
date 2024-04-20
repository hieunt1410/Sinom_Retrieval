import numpy as np

seed = 2024

arr = np.arange(252)
np.random.seed(seed)
np.random.shuffle(arr)

train = arr[:int(0.8*len(arr))]
valid = arr[int(0.8*len(arr)):int(0.9*len(arr))]
test = arr[int(0.9*len(arr)):]

train.to_csv('data/train.csv', index=False)
valid.to_csv('data/valid.csv', index=False)
test.to_csv('data/test.csv', index=False)