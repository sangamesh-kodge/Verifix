
from collections import OrderedDict

# Reading synsets.txt to obtain (ImageNet Class Names). read the file to a dictionary. First column is the key, second column is the value
imagenet_names = OrderedDict()
with open('./info/synsets.txt', 'r') as f:
    for line in f:
        line = line.strip()
        key, value = line.split(' ')[0], " ".join(line.split(' ')[1:])
        imagenet_names[key] = value

# read the file to a list, which is the 1000 classes of imagenet
Imagenet_keys = list(imagenet_names.keys())


# Reading queries_synsets_map.txt (WebVision Folder Mapping). Read the file to a dictionary. First column is the key, second column is the value
folder_class = OrderedDict()
with open('./info/queries_synsets_map.txt', 'r') as f:
    for line in f:
        line = line.strip()
        key, value = line.split(' ')
        folder_class[key] = value

# Creating imagenet_folder_name.txt
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
    web_img[folder_name] = Imagenet_keys[int(class_number)-1]
  
with open('./info/imagenet_folder_name.txt', 'w') as f:
    #  Writes the web_img dictionary to the file
    for key, value in web_img.items():
        f.write(f"{key} {value}\n")


# Reading val_filelist.txt (Validation Image Mapping). read the file to a dictionary. First column is the key, second column is the value
val_list = OrderedDict()
with open('./info/val_filelist.txt', 'r') as f:
    for line in f:
        line = line.strip()
        key, value = line.split(' ')
        val_list[key] = value

# Creating val_imagenet_class.txt.
val_img_class = dict()
for key, value in val_list.items():
    val_img_class[key] = Imagenet_keys[int(value)]

with open('./info/val_imagenet_class.txt', 'w') as f:
    # Writes the val_img_class dictionary to the file
    for key, value in val_img_class.items():
        f.write(f"{key} {value}\n")


