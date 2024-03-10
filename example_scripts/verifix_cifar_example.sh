# DEVICE=0 DATASET=CIFAR10 EPOCH=350 ARCH=ResNet18 BATCH_SIZE=64 sh ./scripts/verifix_cifar_example.sh
# DEVICE=0 DATASET=CIFAR100 EPOCH=350 ARCH=ResNet18 BATCH_SIZE=64 sh ./scripts/verifix_cifar_example.sh
data_path=./data/$DATASET
val_index_path=./pretrained_models/$DATASET
percentage_mislabeled=0.25

# VANILLA
load_loc=./pretrained_models/$DATASET/$DATASET'_final'/train-$ARCH-MisLabeled$percentage_mislabeled
for seed in 42
do 
    for arch in  $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 5e-4
            do 
                for bsz in $BATCH_SIZE
                do 
                    CUDA_VISIBLE_DEVICES=$DEVICE python ./main.py  \
                    --data-path $data_path --dataset $DATASET --use-valset --use-curvature --arch $arch --seed $seed \
                    --batch-size $bsz --test-batch-size $((2*bsz)) --max-batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                    --val-index-path $val_index_path  --percentage-mislabeled $percentage_mislabeled \
                    --load-loc $load_loc --model-name-subscript final \
                    --retain-samples 1000 --mode "baseline,sap" --mode-forget "baseline,sap" \
                    --projection-location "pre" --projection-type "baseline,Mr" \
                    --scale-coff "10000,30000,100000,300000,1000000" --scale-coff-forget "1"
                done
            done
        done
    done
done

#SAM
sam_rho=0.05
load_loc=./pretrained_models/$DATASET/$DATASET'_final'/train-$ARCH-MisLabeled$percentage_mislabeled-SAM$sam_rho
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
                    CUDA_VISIBLE_DEVICES=$DEVICE python ./main.py  \
                    --data-path $data_path --dataset $DATASET --use-valset --use-curvature --arch $arch --seed $seed \
                    --batch-size $bsz --test-batch-size $((2*bsz)) --max-batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                    --val-index-path $val_index_path  --percentage-mislabeled $percentage_mislabeled \
                    --load-loc $load_loc --model-name-subscript final \
                    --retain-samples 1000 --mode "baseline,sap" --mode-forget "baseline,sap" \
                    --projection-location "pre" --projection-type "baseline,Mr" \
                    --scale-coff "10000,30000,100000,300000,1000000" --scale-coff-forget "1" --sam-rho $sam_rho 
                done
            done
        done
    done
done