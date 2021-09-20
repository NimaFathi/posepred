import logging
import time
from logging import config

import pandas as pd
import torch

from utils.others import pose_from_vel
from utils.save_load import save_test_results

config.fileConfig('configs/logging.conf')
logger = logging.getLogger('predLogger')


class Predictor:
    def __init__(self, model, dataloader, is_interactive, save_dir):
        self.model = model
        self.dataloader = dataloader
        self.is_interactive = is_interactive
        self.save_dir = save_dir
        self.device = torch.device('cuda')

        self.result = pd.DataFrame()
        self.pred_pose = torch.Tensor().to('cuda')
        self.pred_vel = torch.Tensor().to('cuda')
        self.pred_mask = torch.Tensor().to('cuda')

    def predict(self):
        logger.info("Prediction started ...")
        self.model.eval()
        time0 = time.time()
        self.__predict()
        save_test_results(self.result, [self.pred_pose, self.pred_vel, self.pred_mask], self.save_dir)
        logger.info("-" * 100)
        logger.info('Testing is completed in: %.2f' % (time.time() - time0))

    def __predict(self):
        for data in self.dataloader:
            for i, d in enumerate(data):
                data[i] = d.to(self.device)
            if self.model.args.use_mask:
                obs_pose, obs_vel, obs_mask = data
            else:
                obs_pose, obs_vel = data

            with torch.no_grad():
                outputs = self.model(data)
                pred_vel = outputs[0]
                pred_pose = pose_from_vel(pred_vel, obs_pose[..., -1, :])
                pred_mask = outputs[1] if self.model.args.use_mask else None
                self.store_results(pred_pose, pred_vel, pred_mask)

    def store_results(self, pred_pose, pred_vel, pred_mask):
        # update tensors
        self.pred_pose = torch.cat((self.pred_pose, pred_pose), 0)
        self.pred_vel = torch.cat((self.pred_vel, pred_vel), 0)
        if self.model.args.use_mask:
            self.pred_mask = torch.cat((self.pred_mask, pred_mask), 0)

        # update dataframe
        for i in range(pred_pose.shape[0]):
            if self.model.args.use_mask:
                single_data = {'pred_pose': str(pred_pose[i].detach().cpu().numpy().tolist()),
                               'pred_vel': str(pred_vel[i].detach().cpu().numpy().tolist()),
                               'pred_mask': str(pred_mask[i].detach().cpu().numpy().round().tolist())}
            else:
                single_data = {'pred_pose': str(pred_pose[i].detach().cpu().numpy().tolist()),
                               'pred_vel': str(pred_vel[i].detach().cpu().numpy().tolist())}
            self.result = self.result.append(single_data, ignore_index=True)
