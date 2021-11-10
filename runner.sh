TASK='comp_pred'
JOINT_NAME='neck'
JOINT_NUM='2'
EPOCHS=5
PRED_W=1
COMP_W=1
INDEX=10
SHOWING="observed-completed-future-predicted"
TRAIN_DATASET="human36_walking/10_10_1_new/train"
VALID_DATASET="human36_walking/10_10_1_new/validation"
TEST_DATASET="human36_walking/10_10_1_new/test"
DEVICE="cpu"


echo 'start training.' &&
python -m api.train model=pv_lstm_noisy keypoint_dim=3 train_dataset=$TRAIN_DATASET valid_dataset=$VALID_DATASET epochs=$EPOCHS data.shuffle=true is_noisy=true use_dct=true model.loss.pred_weight=$PRED_W model.loss.pred_weight=$COMP_W data.noise_keypoint=$JOINT_NUM hydra.run.dir="outputs/${JOINT_NAME}/${TASK}/train" device=$DEVICE &&
echo 'finish training.' &&

echo 'start noisefree evaluation.' &&
python -m api.evaluate model=pv_lstm_noisy keypoint_dim=3 dataset=$TEST_DATASET is_noisy=true data.shuffle=True rounds_num=1 load_path="/home/fathi/posepred/outputs/${JOINT_NAME}/${TASK}/train/snapshots/${EPOCHS}.pt" hydra.run.dir="outputs/${JOINT_NAME}/${TASK}/test/noisefree" data.noise_rate=0 device=$DEVICE &&
echo 'finish noisefree evaluation.' &&

echo 'start noisy evaluation.' &&
python -m api.evaluate model=pv_lstm_noisy keypoint_dim=3 dataset=$TEST_DATASET is_noisy=true data.shuffle=True rounds_num=1 load_path="/home/fathi/posepred/outputs/${JOINT_NAME}/${TASK}/train/snapshots/${EPOCHS}.pt" hydra.run.dir="outputs/${JOINT_NAME}/${TASK}/test/noisy" data.noise_keypoint=$JOINT_NUM device=$DEVICE &&
echo 'finish noisy evaluation.' &&

echo 'start noisefree visualization.' &&
python -m api.visualize model='pv_lstm_noisy' keypoint_dim=3  dataset=$TEST_DATASET load_path="/home/fathi/posepred/outputs/${JOINT_NAME}/${TASK}/train/snapshots/${EPOCHS}.pt" dataset_type='jta' pred_frames_num=60 index=$INDEX showing=$SHOWING is_noisy=true hydra.run.dir="outputs/${JOINT_NAME}/${TASK}/vis/noisefree" data.noise_rate=0 device=$DEVICE &&
echo 'finish noisefree visualization.' &&

echo 'start noisy evaluation.' &&
python -m api.visualize model='pv_lstm_noisy' keypoint_dim=3  dataset=$TEST_DATASET load_path="/home/fathi/posepred/outputs/${JOINT_NAME}/${TASK}/train/snapshots/${EPOCHS}.pt" dataset_type='jta' pred_frames_num=60 index=$INDEX showing=$SHOWING is_noisy=true hydra.run.dir="outputs/${JOINT_NAME}/${TASK}/vis/noisy" data.noise_keypoint=$JOINT_NUM device=$DEVICE &&
echo 'finish noisy evaluation.' &&

echo 'all done.'