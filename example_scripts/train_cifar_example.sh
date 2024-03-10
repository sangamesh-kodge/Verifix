# DEVICE=0 DATASET=CIFAR10 EPOCH=350 ARCH=ResNet18 BATCH_SIZE=64 sh ./scripts/train_cifar_example.sh
# DEVICE=0 DATASET=CIFAR100 EPOCH=350 ARCH=ResNet18 BATCH_SIZE=64 sh ./scripts/train_cifar_example.sh
data_path=./data/$DATASET
model_path=./pretrained_models/$DATASET
percentage_mislabeled=0.25


# VANILLA
for seed in 42
do 
    for arch in $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 5e-4
            do 
                for bsz in $BATCH_SIZE
                do 

                    CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                    --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 --percentage-mislabeled $percentage_mislabeled \
                    --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH
                done
            done
        done
    done
done


# SAM
sam_rho=0.05
for seed in 42
do 
    for arch in $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 5e-4
            do 
                for bsz in $BATCH_SIZE
                do 

                    CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                    --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 --percentage-mislabeled $percentage_mislabeled \
                    --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH --sam-rho $sam_rho
                done
            done
        done
    done
done