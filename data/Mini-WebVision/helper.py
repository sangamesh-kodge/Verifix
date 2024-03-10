# read a text file as dict. First column is key, second column is value
import json
from collections import OrderedDict

# read the file to a dictionary. First column is the key, second column is the value
imagenet_names = OrderedDict()
with open('./info/synsets.txt', 'r') as f:
    for line in f:
        line = line.strip()
        key, value = line.split(' ')[0], " ".join(line.split(' ')[1:])
        imagenet_names[key] = value

# read the file to a list, which is the 1000 classes of imagenet
Imagenet_keys = list(imagenet_names.keys())


# read the file to a dictionary. First column is the key, second column is the value
folder_class = OrderedDict()
with open('./info/queries_synsets_map.txt', 'r') as f:
    for line in f:
        line = line.strip()
        key, value = line.split(' ')
        folder_class[key] = value

# create a dictionary. The key is the folder name, value is the class name.
# The folder names are in format q0001 to q9983, but we need to add the q to make it correct.
web_img = dict()
for folder, class_number in folder_class.items():
    if int(folder)<10:
        folder_name = f"q000{folder}"
    elif int(folder)<100:
        folder_name = f"q00{folder}"
    elif int(folder)<1000:
        folder_name = f"q0{folder}"
    else:
        folder_name = f"q{folder}"
    # only add to the dictionary if the class number is in the range of 1 to 50.
    if int(class_number)<=50:
        web_img[folder_name] = Imagenet_keys[int(class_number)-1]

# Save web_img dict as txt file    
with open('./info/imagenet_folder_name.txt', 'w') as f:
    for key, value in web_img.items():
        f.write(f"{key} {value}\n")


# read the file to a dictionary. First column is the key, second column is the value
val_list = OrderedDict()
with open('./info/val_filelist.txt', 'r') as f:
    for line in f:
        line = line.strip()
        key, value = line.split(' ')
        val_list[key] = value

# create a dictionary. The key is the val file name, value is the class name.
val_img_class = dict()
for key, value in val_list.items():
    # only add to the dictionary if the class number is in the range of 1 to 50.
    if int(value)<50:
        val_img_class[key] = Imagenet_keys[int(value)]
# Save val_img_class dict as txt file
with open('./info/val_imagenet_class.txt', 'w') as f:
    for key, value in val_img_class.items():
        f.write(f"{key} {value}\n")


