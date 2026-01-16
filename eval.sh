python classification_experiment_multimethod.py \
    --calibration-dir /mnt/nas_d/data/deepfake/collected_dataset/ForenSynths/test/cyclegan/winter/0_real \
    --test-dir ~/.cache/kagglehub/datasets/yangsangtai/tiny-genimage/versions/1/ \
    --criterion score_laplacian_fast \
    --num-noise 16 \
    --batch-size 2 \
    --output-dir results


