# DEVICE=0 DATASET=Mini-WebVision EPOCH=200 ARCH=InceptionResNetV2 BATCH_SIZE=64 sh ./scripts/train_realnoise_example.sh
# DEVICE=0 DATASET=WebVision1.0 EPOCH=60 ARCH=InceptionResNetV2 BATCH_SIZE=512 sh ./scripts/train_realnoise_example.sh
# DEVICE=0 DATASET=Clothing1M EPOCH=20 ARCH=ResNet50 BATCH_SIZE=512 sh ./scripts/train_realnoise_example.sh

## For Clothing1M loading pretrained model. (download from https://download.pytorch.org/models/resnet50-0676ba61.pth) 
## Uncomment the following 3 lines.
# mkdir -p ./pretrained_models/ImageNet1k/
# wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O ./pretrained_models/ImageNet1k/torchvision_resnet50.pt
# PRETRAIN_PATH=./pretrained_models/ImageNet1k/torchvision_resnet50.pt

data_path=./data/$DATASET
model_path=./pretrained_models/$DATASET

# VANILLA
for seed in 42
do 
    for arch in $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 4e-5
            do 
                for bsz in $BATCH_SIZE
                do 
                    if [[ -z "${PRETRAIN_PATH}" ]]; then
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH
                    else
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                        --load-loc $PRETRAIN_PATH
                    fi
                done
            done
        done
    done
done

# MIXUP
mixup_alpha=0.2
for seed in 42
do 
    for arch in $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 4e-5
            do 
                for bsz in $BATCH_SIZE
                do 
                    if [[ -z "${PRETRAIN_PATH}" ]]; then
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH --mixup-alpha $mixup_alpha
                    else
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                        --load-loc $PRETRAIN_PATH  --mixup-alpha $mixup_alpha
                    fi
                done
            done
        done
    done
done


# MentorMix
mnet_gamma_p=0.85
mmix_alpha=0.2
for seed in 42
do 
    for arch in $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 4e-5
            do 
                for bsz in $BATCH_SIZE
                do 
                    if [[ -z "${PRETRAIN_PATH}" ]]; then
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH --mnet-gamma-p $mnet_gamma_p --mmix-alpha $mixup_alpha
                    else
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                        --load-loc $PRETRAIN_PATH  --mnet-gamma-p $mnet_gamma_p --mmix-alpha $mmix_alpha
                    fi
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
            for wd in 4e-5
            do 
                for bsz in $BATCH_SIZE
                do 
                    if [[ -z "${PRETRAIN_PATH}" ]]; then
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH --sam-rho $sam_rho
                    else
                        CUDA_VISIBLE_DEVICES=$DEVICE python ./train.py --dataset $DATASET --data-path $data_path \
                        --arch $arch --seed $seed --model-path $model_path --use-valset  0.1 \
                        --batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                        --load-loc $PRETRAIN_PATH  --sam-rho $sam_rho
                    fi
                done
            done
        done
    done
done
