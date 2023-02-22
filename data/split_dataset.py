import pickle
import os

data_file = 'yoochoose/yoochoose-filter-new.pkl'

with open(data_file, 'rb') as f:
    dataset = pickle.load(f)

dataset_len = len(list(dataset.keys()))
print(1/4)
# s
for s in [4, 16]:

    sub_len = int(dataset_len*(1/s))
    print(sub_len)
    if not os.path.exists(f'{s}yoochoose/'):
        os.makedirs(f'{s}yoochoose/')  
    sub_dataset = dict(list(dataset.items())[:sub_len])
    with open(f'{s}yoochoose/{s}yoochoose-filter-new.pkl', 'wb') as f:
        pickle.dump(sub_dataset, f)

print('done!')

