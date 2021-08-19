import time
import torch
from utils.others import pose_from_vel


class Tester:
    def __init__(self, model, dataloader, is_interactive):
        self.model = model
        self.dataloader = dataloader
        self.is_interactive = is_interactive
        self.device = torch.device('cuda')

    def test(self):
        self.model.eval()
        time0 = time.time()
        self.test_()
        print("-" * 100)
        print('Testing is completed in: %.2f' % (time.time() - time0))

    def test_(self):
        for data in self.dataloader:
            if self.is_interactive:
                pass
            else:
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
                    if self.model.args.use_mask:
                        pred_mask = outputs[1]
