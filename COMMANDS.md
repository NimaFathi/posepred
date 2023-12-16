Here you can find the results of several models on Human3.6M:

| **Model**                  | **$80 ms$** | **$160 ms$** | **$320 ms$** | **$400 ms$** | **$560 ms$** | **$720 ms$** | **$880 ms$** | **$1000 ms$** |
|----------------------------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|
| **Zero-Vel**               | 23.8        | 44.4         | 76.1         | 88.2         | 107.4        | 121.6        | 131.6        | 136.6         |
| **MSR-GCN**                | 12.0        | 25.2         | 50.4         | 61.4         | 80.0         | 93.9         | 105.5        | 112.9         |
| **STS-GCN**                | 17.7        | 33.9         | 56.3         | 67.5         | 85.1         | 99.4         | 109.9        | 117.0         |
| **STS-GCN + pUAL (ours)**  | 13.2        | 27.1         | 54.7         | 66.2         | 84.5         | 97.9         | 109.3        | 115.7         |
| **HRI\***                   | 12.7        | 26.1         | 51.5         | 62.6         | 80.8         | 95.1         | 106.8        | 113.8         |
| **HRI\* + pUAL (ours)**     | 11.6        | 25.3         | 51.2         | 62.2         | 80.1         | 93.7         | 105.0        | 112.1         |
| **PGBIG**                  | 10.3        | 22.6         | 46.6         | 57.5         | 76.3         | 90.9         | 102.7        | 110.0         |
| **PGBIG + pUAL (ours)**    | 9.6         | 21.7         | 46.0         | 57.1         | 75.9         | 90.3         | 102.1        | 109.5         |
| **ST-Trans**               | 13.0        | 27.0         | 52.6         | 63.2         | 80.3         | 93.6         | 104.7        | 111.6         |
| **ST-Trans + pUAL (ours)** | 10.4        | 23.4         | 48.4         | 59.2         | 77.0         | 90.7         | 101.9        | 109.3         |


Similarly on AMASS and 3DPW datasets

Using the commands below you can train different models on different datasets. 

**NOTE**: AMASS and 3DPW settings are simillar to each other.
# ST_Trans
## Human3.6M
```bash
python -m api.train model=st_trans \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=50 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=human3.6m \
    model.n_major_joints=22 \
    model.loss.nJ=32\
    epochs=15
```
## AMASS
```bash
python -m api.train model=st_trans \
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
python -m api.train model=st_trans \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.loss.nT=25 \
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
    model.loss.pre_post_process=human3.6m \
    epochs=50
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
    model.loss.pre_post_process=AMASS \
    epochs=50
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
    model.loss.pre_post_process=3DPW \
    epochs = 50
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