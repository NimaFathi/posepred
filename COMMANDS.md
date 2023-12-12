**NOTE**: AMASS and 3DPW settings are simillar to each other.
# ST_Transformer
## Human3.6M
```bash
python -m api.train model=st_transformer \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=human3.6m \
    model.n_major_joints=22 \
    model.loss.nJ=32
```
## AMASS
```bash
python -m api.train model=st_transformer \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=AMASS \
    model.n_major_joints=18 \
    model.loss.nJ=18 
```
## 3DPW
```bash
python -m api.train model=st_transformer \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=3DPW \
    model.n_major_joints=18 \
    model.loss.nJ=18
```

# History-Repeats-Itself
## Human3.6M
```bash
python -m api.train model=history_repeats_itself \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    model.modality=Human36 \
    model.in_features=66 \
    obs_frames_num=50 \
    pred_frames_num=25
```
## AMASS
```bash
python -m api.train model=history_repeats_itself \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    model.modality=AMASS \
    model.in_features=66 \
    obs_frames_num=50 \
    pred_frames_num=25
```

## 3DPW
```bash
python -m api.train model=history_repeats_itself \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    model.modality=3DPW \
    model.in_features=66 \
    obs_frames_num=50 \
    pred_frames_num=25
```

# STS-GCN
## Human3.6M
```bash
python -m api.train model=sts_gcn \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=human3.6m \
    model.n_major_joints=22 \
    model.loss.nJ=32
```
## AMASS
```bash
python -m api.train model=sts_gcn \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=AMASS \
    model.n_major_joints=18 \
    model.loss.nJ=18
```

## 3DPW
```bash
python -m api.train model=sts_gcn \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=3DPW \
    model.n_major_joints=18 \
    model.loss.nJ=18
```
# PGBIG
## Human3.6M
```bash
python -m api.train model=pgbig \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=human3.6m \
    model.in_features=66 \
    model.loss.nJ=22 \
    model.loss.pre_post_process=human3.6m
```
## AMASS
```bash
python -m api.train model=pgbig \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=AMASS \
    model.in_features=54 \
    model.loss.nJ=18 \
    model.loss.pre_post_process=AMASS
```
## 3DPW
```bash
python -m api.train model=pgbig \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=3DPW \
    model.in_features=54 \
    model.loss.nJ=18 \
    model.loss.pre_post_process=3DPW
```
# POTR
```bash
python -m api.train model=potr \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    keypoint_dim=9 \
    model_pose_format=rotmat \
    metric_pose_format=euler \
    obs_frames_num=16 \
    pred_frames_num=12 \
    pose_metrics=[MSE]
```
**NOTE**: POTR model only works when the datset is in the format of `rotmat` and the metric is `euler`. This setting is only available in the `Human36` dataset.

# MSR-GCN
```bash
python -m api.train model=msr_gcn \
    train_dataset=$DATASET_VALIDATION_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25
```

# PV-LSTM
```bash
python -m api.train model=pv_lstm \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    model.loss.nJ=32 \
    obs_frames_num=10 \
    pred_frames_num=25
```

# Derpof
```bash
python -m api.train model=derpof \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=50 \
    pred_frames_num=25 \
    data.seq_rate=5
```

# Disentangled

```bash
python -m api.train model=disentangled \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25
```