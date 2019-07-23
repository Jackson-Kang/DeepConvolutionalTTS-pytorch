import csv
import random
import numpy as np


f = open('/home/minsu/data/LJSpeech-1.1/metadata.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

data_len = 0
for _ in rdr:
        data_len += 1


index_list = [idx for idx in range(data_len)]

split_ratio = 0.1

num_of_test_and_valid_dataset = int(data_len*split_ratio)	
index_of_test_and_valid_dataset = random.sample(index_list, num_of_test_and_valid_dataset)

random.shuffle(index_of_test_and_valid_dataset)

split_length = int(len(index_of_test_and_valid_dataset)/2)
valid_data_idx = index_of_test_and_valid_dataset[:split_length]
test_data_idx = index_of_test_and_valid_dataset[split_length:]

check = 0


for i in test_data_idx:
	for j in valid_data_idx:
		if i== j :
			print('ARE YOU CRAZY?????! Duplicated Values exist!')
			check = 1


if check ==0:
	print(len(valid_data_idx), len(test_data_idx))

	np.save('./lj_eval_idx.npy', valid_data_idx)
	np.save('./lj_test_idx.npy', test_data_idx)

f.close()
