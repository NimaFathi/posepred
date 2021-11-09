echo 'start training.'
CUDA_VISIBLE_DEVICES=1 python -m api.train model=pv_lstm_noisy keypoint_dim=3 train_dataset='JTA/3D/train_16_60_0_JTA' valid_dataset='JTA/3D/validation_16_60_0_JTA' epochs=200 data.shuffle=true is_noisy=true use_dct=true model.loss.pred_weight=0 data.noise_keypoint=2 hydra.run.dir=outputs/neck/comp/train
echo 'finish training.'

echo 'start noisefree evaluation.'
CUDA_VISIBLE_DEVICES=1 python -m api.evaluate model=pv_lstm_noisy keypoint_dim=3 dataset='JTA/3D/test_16_60_0_JTA' is_noisy=true data.shuffle=True rounds_num=1 load_path='/home/fathi/posepred/outputs/neck/comp/train/snapshots/200.pt' hydra.run.dir=outputs/neck/comp/test/noisefree data.noise_rate=0
echo 'finish noisefree evaluation.'

echo 'start noisy evaluation.'
CUDA_VISIBLE_DEVICES=1 python -m api.evaluate model=pv_lstm_noisy keypoint_dim=3 dataset='JTA/3D/test_16_60_0_JTA' is_noisy=true data.shuffle=True rounds_num=1 load_path='/home/fathi/posepred/outputs/neck/comp/train/snapshots/200.pt' hydra.run.dir=outputs/neck/comp/test/noisy data.noise_keypoint=2
echo 'finish noisy evaluation.'

echo 'start noisefree visualization.'
CUDA_VISIBLE_DEVICES=1 python -m api.visualize model='pv_lstm_noisy' keypoint_dim=3  dataset='JTA/3D/test_16_60_0_JTA' load_path='/home/fathi/posepred/outputs/neck/comp/train/snapshots/200.pt' dataset_type='jta' pred_frames_num=60 index=10 showing='observed-completed' is_noisy=true hydra.run.dir=outputs/neck/comp/vis/noisefree data.noise_rate=0
echo 'finish noisefree visualization.'

echo 'start noisy evaluation.'
CUDA_VISIBLE_DEVICES=1 python -m api.visualize model='pv_lstm_noisy' keypoint_dim=3  dataset='JTA/3D/test_16_60_0_JTA' load_path='/home/fathi/posepred/outputs/neck/comp/train/snapshots/200.pt' dataset_type='jta' pred_frames_num=60 index=10 showing='observed-completed' is_noisy=true hydra.run.dir=outputs/neck/comp/vis/noisy data.noise_keypoint=2
echo 'finish noisy evaluation.'

echo 'all done.'