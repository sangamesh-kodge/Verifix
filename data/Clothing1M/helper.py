# get clean kv for clean train, clean val and clean test
from collections import defaultdict


with open('clean_label_kv.txt') as f:
    clean_label_kv = f.readlines()
clean_label_kv_dict = defaultdict(None)
for line in clean_label_kv:
    line = line.split("\n")[0]
    folder_name, class_number = line.strip().split(" ")
    clean_label_kv_dict[folder_name] = class_number


with open('clean_val_key_list.txt', 'r') as f:
    clean_val_k = f.readlines()
clean_val_kv=defaultdict(None)
for key in clean_val_k:
    key = key.split("\n")[0]
    clean_val_kv[key] = clean_label_kv_dict[key]
with open("clean_val_kv.txt", "w") as f:
    for k,v in clean_val_kv.items():
        f.write(f"{k} {v}\n")

with open('clean_test_key_list.txt', 'r') as f:
    clean_test_k = f.readlines()
clean_test_kv=defaultdict(None)  
for key in clean_test_k:
    key = key.split("\n")[0]
    clean_test_kv[key] = clean_label_kv_dict[key]
with open("clean_test_kv.txt", "w") as f:
    for k,v in clean_test_kv.items():
        f.write(f"{k} {v}\n")

with open('clean_train_key_list.txt', 'r') as f:
    clean_train_k = f.readlines()
clean_train_kv=defaultdict(None)
for key in clean_train_k:
    key = key.split("\n")[0]
    clean_train_kv[key] = clean_label_kv_dict[key]
with open("clean_train_kv.txt", "w") as f:
    for k,v in clean_train_kv.items():
        if (k not in clean_test_kv) and (k not in clean_val_kv):
            f.write(f"{k} {v}\n")


with open('noisy_label_kv.txt', 'r') as f:
    noisy_label_kv = f.readlines()
noisy_label_kv_dict = defaultdict(None)
for line in noisy_label_kv:
    line = line.split("\n")[0]
    image_path, class_number = line.strip().split(" ")
    noisy_label_kv_dict[image_path] = class_number
with open('noisy_train_key_list.txt', 'r') as f:
    noisy_train_k = f.readlines()
noisy_train_kv=defaultdict(None)
for key in noisy_train_k:
    key = key.split("\n")[0]
    noisy_train_kv[key] = noisy_label_kv_dict[key]
with open("noisy_train_kv.txt", "w") as f:
    for k,v in noisy_train_kv.items():
        if (k not in clean_test_kv) and (k not in clean_val_kv):
            f.write(f"{k} {v}\n")


