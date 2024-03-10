### Download files
# Download necessary tar files from the WebVision dataset
wget https://data.vision.ee.ethz.ch/cvl/webvision/google_resized_256.tar
wget https://data.vision.ee.ethz.ch/cvl/webvision/flickr_resized_256.tar
wget https://data.vision.ee.ethz.ch/cvl/webvision/val_images_256.tar
wget https://data.vision.ee.ethz.ch/cvl/webvision/info.tar

### Make directory and move tar files
# Create 'train' and 'val' directories to hold training and validation data
mkdir ./train
mkdir ./val
# Move the downloaded tar files into the 'train' and 'val' directories
mv ./google_resized_256.tar ./train/
mv ./flickr_resized_256.tar ./train/
mv ./val_images_256.tar ./val/

### Move and uncompress the train files
cd ./train
tar -xf google_resized_256.tar
tar -xf flickr_resized_256.tar
cd ../

### Move and uncompress the val files
cd ./val
tar -xf val_images_256.tar
cd ..

### Move and uncompress the info files
tar -xf info.tar

### Make helper files using helper.py python file.
python helper.py

### Copy train images to respective folder
echo "----------------------------------------------------------------"
echo "Creating directory structure similar to ImageNet for training dataset"
echo "----------------------------------------------------------------"
input="./info/imagenet_folder_name.txt"
path_folder="./train"
while IFS= read -r line
do
    web_folder_name=${line:0:5}
    img_folder_name=${line:6}
    mkdir -p $path_folder/$img_folder_name
    mv $path_folder/google/$web_folder_name/* $path_folder/$img_folder_name/
    mv $path_folder/flickr/$web_folder_name/* $path_folder/$img_folder_name/    
done < "$input"

### Copy val images to respective folder
echo "----------------------------------------------------------------"
echo "Creating directory structure similar to ImageNet for val dataset"
echo "----------------------------------------------------------------"
input="./info/val_imagenet_class.txt"
path_folder="./val"
while IFS= read -r line
do
    imagenet_name=${line:0:13}
    img_folder_name=${line:14}
    mkdir -p $path_folder/$img_folder_name
    mv $path_folder/val_images_256/$imagenet_name $path_folder/$img_folder_name/
done < "$input"



# remove files - cleanup process
echo "----------------------------------------------------------------"
echo "Removing Redundant files."
echo "----------------------------------------------------------------"
rm -rf ./info.tar 
rm -rf info/
rm -rf ./val/val_images_256.tar 
rm -r val/val_images_256/
rm -rf ./train/google_resized_256.tar
rm -rf train/google/
rm -rf ./train/flickr_resized_256.tar
rm -rf train/flickr/


echo "----------------------------------------------------------------"
echo "WebVision Dataset 1.0 Processed!"
echo "----------------------------------------------------------------"
