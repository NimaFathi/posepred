import logging
import time

import pandas as pd
import torch

from utils.save_load import save_test_results

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, dataloader, is_interactive, save_dir):
        self.model = model
        self.dataloader = dataloader
        self.is_interactive = is_interactive
        self.save_dir = save_dir
        self.device = torch.device('cuda')

        self.result = pd.DataFrame()
        self.pred_pose = torch.Tensor().to('cuda')
        self.pred_mask = torch.Tensor().to('cuda')

    def predict(self):
        logger.info("Prediction started.")
        self.model.eval()
        time0 = time.time()
        self.__predict()
        save_test_results(self.result, [self.pred_pose, self.pred_vel, self.pred_mask], self.save_dir)
        logger.info("-" * 100)
        logger.info('Testing is completed in: %.2f' % (time.time() - time0))

    def __predict(self):
        for data in self.dataloader:
            for key, value in data.items():
                data[key] = value.to(self.device)

            with torch.no_grad():
                # predict & calculate loss
                model_outputs = self.model(data)
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
                pred_pose = model_outputs['pred_pose']

                if self.model.args.use_mask:
                    assert 'pred_mask' in model_outputs.keys(), 'outputs of model should include pred_mask'
                    pred_mask = model_outputs['pred_mask']
                else:
                    pred_mask = None

                self.store_results(pred_pose, pred_mask)

    def store_results(self, pred_pose, pred_mask):
        # update tensors
        self.pred_pose = torch.cat((self.pred_pose, pred_pose), 0)
        if self.model.args.use_mask:
            self.pred_mask = torch.cat((self.pred_mask, pred_mask), 0)

        # update dataframe
        for i in range(pred_pose.shape[0]):
            if self.model.args.use_mask:
                single_data = {'pred_pose': str(pred_pose[i].detach().cpu().numpy().tolist()),
                               'pred_mask': str(pred_mask[i].detach().cpu().numpy().round().tolist())}
            else:
                single_data = {'pred_pose': str(pred_pose[i].detach().cpu().numpy().tolist()), }
            self.result = self.result.append(single_data, ignore_index=True)
