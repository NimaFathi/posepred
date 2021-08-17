from args.dataloader_args import DataloaderArgs
from args.training_args import TrainingArgs
from args.model_args import ModelArgs
from train.trainer import Trainer
from data_loader.data_loader import get_dataloader

if __name__ == '__main__':
    dataloader_args = DataloaderArgs('simple_dataset', data_dim=2, is_testing=False, use_mask=False, skip_frame=0,
                                     batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    training_args = TrainingArgs(epochs=6, start_epoch=0, lr=0.001, decay_factor=0.95, decay_patience=20,
                                 distance_loss='L1', mask_loss_weight=0.25, snapshot_interval=2)

    model_args = ModelArgs(model_name='lstm_vel', hidden_size=200, hardtanh_limit=10, n_layers=1, dropout_enc=0,
                           dropout_pose_dec=0, dropout_mask_dec=0)

    trainer = Trainer(training_args, dataloader_args, dataloader_args, model_args)
    print(trainer.dim)

    # dataloader = get_dataloader(dataloader_args)
    # for data in dataloader:
    #     obs_pose = data[0]
    #     print(obs_pose.shape)
