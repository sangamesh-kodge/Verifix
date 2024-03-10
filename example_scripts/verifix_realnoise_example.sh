
# DEVICE=0 DATASET=Mini-WebVision EPOCH=200 ARCH=InceptionResNetV2 BATCH_SIZE=128 sh ./scripts/verifix_realnoise_example.sh
# DEVICE=0 DATASET=WebVision1.0 EPOCH=60 ARCH=InceptionResNetV2 BATCH_SIZE=128 sh ./scripts/verifix_realnoise_example.sh
# DEVICE=0 DATASET=Clothing1M EPOCH=20 ARCH=ResNet50 BATCH_SIZE=128 sh ./scripts/verifix_realnoise_example.sh
data_path=./data/$DATASET
val_index_path=./pretrained_models/$DATASET
percentage_mislabeled=0.0

# VANILLA
load_loc=./pretrained_models/$DATASET/$DATASET'_final'/train-$ARCH-MisLabeled$percentage_mislabeled
for seed in  42
do 
    for arch in  $ARCH
    do
        for lr in 1e-2
        do 
            for wd in 4e-5
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

# MIXUP
mixup_alpha=0.2
load_loc=./pretrained_models/$DATASET/$DATASET'_final'/train-$ARCH-MisLabeled$percentage_mislabeled-MixUp$mixup_alpha
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
                    CUDA_VISIBLE_DEVICES=$DEVICE python ./main.py  \
                    --data-path $data_path --dataset $DATASET --use-valset --use-curvature --arch $arch --seed $seed \
                    --batch-size $bsz --test-batch-size $((2*bsz)) --max-batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                    --val-index-path $val_index_path  \
                    --load-loc $load_loc --model-name-subscript final \
                    --retain-samples 1000 --mode "baseline,sap" --mode-forget "baseline,sap" \
                    --projection-location "pre" --projection-type "baseline,Mr" \
                    --scale-coff "1000,3000,10000,30000,100000,300000,1000000" --scale-coff-forget "1" --mixup-alpha $mixup_alpha
                done
            done
        done
    done
done

# MentorMix
mnet_gamma_p=0.85
mmix_alpha=0.2
load_loc=./pretrained_models/$DATASET/$DATASET'_final'/train-$ARCH-MisLabeled$percentage_mislabeled-MMix$mnet_gamma_p-$mmix_alpha
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
                    CUDA_VISIBLE_DEVICES=$DEVICE python ./main.py  \
                    --data-path $data_path --dataset $DATASET --use-valset --use-curvature --arch $arch --seed $seed \
                    --batch-size $bsz --test-batch-size $((2*bsz)) --max-batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                    --val-index-path $val_index_path  \
                    --load-loc $load_loc --model-name-subscript final \
                    --retain-samples 1000 --mode "baseline,sap" --mode-forget "baseline,sap" \
                    --projection-location "pre" --projection-type "baseline,Mr" \
                    --scale-coff "1000,3000,10000,30000,100000,300000,1000000" --scale-coff-forget "1"  --mnet-gamma-p $mnet_gamma --mmix-alpha $mmix_alpha
                done
            done
        done
    done
done

# SAM
sam_rho=0.05
load_loc=NANO_HOME/label_robustness/WebVision1.0_final-v5/train-InceptionResNetV2-MisLabeled0.0-SAM$sam_rho
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
                    CUDA_VISIBLE_DEVICES=$DEVICE python ./main.py  \
                    --data-path $data_path --dataset $DATASET --use-valset --use-curvature --arch $arch --seed $seed \
                    --batch-size $bsz --test-batch-size $((2*bsz)) --max-batch-size $bsz --weight-decay $wd --lr $lr --epoch $EPOCH \
                    --val-index-path $val_index_path  \
                    --load-loc $load_loc --model-name-subscript final \
                    --retain-samples 1000 --mode "baseline,sap" --mode-forget "baseline,sap" \
                    --projection-location "pre" --projection-type "baseline,Mr" \
                    --scale-coff "1000,3000,10000,30000,100000,300000,1000000" --scale-coff-forget "1" --sam-rho $sam_rho
                done
            done
        done
    done
done