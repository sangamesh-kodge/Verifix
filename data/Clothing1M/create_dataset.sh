
echo "----------------------------------------------------------------"
echo "Make Directory structure"
echo "----------------------------------------------------------------"
mkdir clean_train val test noisy_train
for path_folder in ./val ./test ./clean_train ./noisy_train
do
    for i in {0..13}
    do
        if [ "$i" -gt 9 ];
        then
            move_path="$path_folder/q00$i";
        else
            move_path="$path_folder/q000$i";
        fi;    
        mkdir -p $move_path
    done
done

# echo "----------------------------------------------------------------"
# echo "Unzip Files"
# echo "----------------------------------------------------------------"
# unzip annotations.zip
# cd images/
# for i in {0..9}
# do
#     tar -xf $i.tar
# done
# cd ../

echo "----------------------------------------------------------------"
echo "Creating helper files"
echo "----------------------------------------------------------------"
python helper.py

echo "----------------------------------------------------------------"
echo "Creating clean val directory structure clasewise"
echo "----------------------------------------------------------------"
input="./clean_val_kv.txt"
path_folder="./val"
while IFS= read -ra line
do
    text=( $line )
    folder_name=${text[0]}
    class_number=${text[1]}
    if [ "$class_number" -gt 9 ];
    then
        move_path="$path_folder/q00$class_number";
    else
        move_path="$path_folder/q000$class_number";
    fi;
    mv $folder_name $move_path
done < "$input"




echo "----------------------------------------------------------------"
echo "Creating clean test directory structure clasewise"
echo "----------------------------------------------------------------"
input="./clean_test_kv.txt"
path_folder="./test"
while IFS= read -ra line
do
    text=( $line )
    folder_name=${text[0]}
    class_number=${text[1]}
    if [ "$class_number" -gt 9 ];
    then
        move_path="$path_folder/q00$class_number";
    else
        move_path="$path_folder/q000$class_number";
    fi;
    mv $folder_name $move_path
done < "$input"


echo "----------------------------------------------------------------"
echo "Creating clean train directory structure clasewise"
echo "----------------------------------------------------------------"
input="./clean_train_kv.txt"
path_folder="./clean_train"
while IFS= read -ra line
do
    text=( $line )
    folder_name=${text[0]}
    class_number=${text[1]}
    if [ "$class_number" -gt 9 ];
    then
        move_path="$path_folder/q00$class_number";
    else
        move_path="$path_folder/q000$class_number";
    fi;
    cp $folder_name $move_path
done < "$input"


echo "----------------------------------------------------------------"
echo "Creating noisy train directory structure clasewise"
echo "----------------------------------------------------------------"
input="./noisy_train_kv.txt"
path_folder="./noisy_train"
while IFS= read -ra line
do
    text=( $line )
    folder_name=${text[0]}
    class_number=${text[1]}
    if [ "$class_number" -gt 9 ];
    then
        move_path="$path_folder/q00$class_number";
    else
        move_path="$path_folder/q000$class_number";
    fi;
    cp $folder_name $move_path
done < "$input"

echo "----------------------------------------------------------------"
echo "Dataset Created"
echo "----------------------------------------------------------------"

echo "----------------------------------------------------------------"
echo "Clean Directory "
echo "----------------------------------------------------------------"
rm -r images/
rm -r annotations.zip



