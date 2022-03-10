export HYDRA_FULL_ERROR=1

DATASET_PATH='/home/zahra/workshop/datasets/Poses/Poses'
DATASET_NAME='human3.6m'

# rm /home/zahra/workshop/posepred-v3/posepred/preprocessed_data/human36m/train_10_25_1_human3.6m.jsonl
# rm /home/zahra/workshop/posepred-v3/posepred/preprocessed_data/human36m/train_1_human3.6m.jsonl

python3 -m api.preprocess dataset=$DATASET_NAME \
official_annotation_path=$DATASET_PATH \
skip_num=1 \
obs_frames_num=50 \
pred_frames_num=25 \
keypoint_dim=3 \
data_type=train \
interactive=false


# rm /home/zahra/workshop/posepred-v3/posepred/preprocessed_data/human36m/validation_10_25_1_human3.6m.jsonl
# rm /home/zahra/workshop/posepred-v3/posepred/preprocessed_data/validation_1_human3.6m.jsonl

python3 -m api.preprocess dataset=$DATASET_NAME \
official_annotation_path=$DATASET_PATH \
skip_num=1 \
obs_frames_num=50 \
pred_frames_num=25 \
keypoint_dim=3 \
data_type=validation \
interactive=false

# DATASET_TRAIN_PATH='/home/zahra/workshop/posepred-v3/posepred/preprocessed_data/human36m/train_10_25_1_human3.6m.jsonl'
# DATASET_VALIDATION_PATH='/home/zahra/workshop/posepred-v3/posepred/preprocessed_data/human36m/validation_10_25_1_human3.6m.jsonl'
# OUTPUT_PATH='/home/zahra/workshop/posepred-v3/posepred/outputs/'$(date +"potr_%T")

# python -m api.train model=msr_gcn \
# keypoint_dim=3 \
# train_dataset=$DATASET_TRAIN_PATH \
# valid_dataset=$DATASET_VALIDATION_PATH \
# epochs=2 \
# data.shuffle=True \
# device=cpu \
# snapshot_interval=10 \
# hydra.run.dir=$OUTPUT_PATH \
# data.use_xyz=True
# data.loader=ours

#MODEL_OUTPUT_PATH='/home/helium/Desktop/VITA/posepred/outputs/sts_gcn/snapshots/300.pt'

#python -m api.evaluate model=sts_gcn\
# dataset=$DATASET_VALIDATION_PATH keypoint_dim=3\
#  data.shuffle=True rounds_num=5\
#   load_path=$MODEL_OUTPUT_PATH device=cuda


#python -m api.visualize model=sts_gcn\
# dataset=$DATASET_VALIDATION_PATH dataset_type=$DATASET_NAME keypoint_dim=3\
# device=cuda load_path=$MODEL_OUTPUT_PATH index=125 showing=observed-future-predicted
























