import logging
import time

import torch
import numpy as np
import os
from tqdm import tqdm

from metrics import POSE_METRICS
from utils.others import dict_to_device

logger = logging.getLogger(__name__)

#new:
def visualize_r(model_outputs,data):
    predicted_poses = model_outputs['pred_pose'][0].clone()
    predicted_poses = predicted_poses.reshape(predicted_poses.shape[0],-1,3).cpu().numpy()
    obsereved_poses = data['observed_pose'][0].clone()
    obsereved_poses = obsereved_poses.reshape(obsereved_poses.shape[0],-1,3).cpu().numpy()
    gt_poses = data['future_pose'][0].clone()
    gt_poses = gt_poses.reshape(gt_poses.shape[0],-1,3).cpu().numpy()
    
    KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    predicted_poses = predicted_poses[:,KeyPoints_from3d,:]
    obsereved_poses = obsereved_poses[:,KeyPoints_from3d,:]
    gt_poses = gt_poses[:,KeyPoints_from3d,:]
    skeleton = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    import matplotlib.pyplot as plt
    for i in range(gt_poses.shape[0]):
        xdata = gt_poses[i].T[0]/1000
        ydata = gt_poses[i].T[1]/1000
        zdata = gt_poses[i].T[2]/1000
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(xdata,ydata,zdata, color ="mediumvioletred" , label =data['action'][0])
        for j in range(17):
            ax.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]] , color = "palevioletred" )
        
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.axes.set_zlim3d(bottom=-1 , top=1 )
        
        plt.legend() 
        
        plt.savefig("/home/rh/codes/posepred/my_temp/test2.png")
        plt.show()
        # breakpoint()
#end new

class Evaluator:
    # evaluator = Evaluator(cfg, eval_dataloader, model, loss_module, eval_reporter)
    def __init__(self, args, dataloader, model, loss_module, reporter):
        self.args = args
        self.dataloader = dataloader
        self.model = model.to(args.device)
        self.loss_module = loss_module.to(args.device)
        self.reporter = reporter
        self.pose_metrics = args.pose_metrics
        self.rounds_num = args.rounds_num
        self.device = args.device
        
        #new:
        self.visualize = True 

    def evaluate(self):
        logger.info('Evaluation started.')
        self.model.eval()
        # self.loss_module.eval()
        for i in range(self.rounds_num):
            logger.info('round ' + str(i + 1) + '/' + str(self.rounds_num))
            self.__evaluate()
        self.reporter.print_pretty_metrics(logger, self.pose_metrics)
        self.reporter.save_csv_metrics(self.pose_metrics, os.path.join(self.args.save_dir,"eval.csv"))
        logger.info("Evaluation has been completed.")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        pose_key = None
        for data in tqdm(self.dataloader):
            actions = set(data['action']) if 'action' in data.keys() else set()
            actions.add("all")
            # TODO
            if pose_key is None:
                pose_key = [k for k in data.keys() if "pose" in k][0]
            batch_size = data[pose_key].shape[0]
            with torch.no_grad():
                # predict & calculate loss
                model_outputs = self.model(dict_to_device(data, self.device))
                loss_outputs = self.loss_module(model_outputs, dict_to_device(data, self.device))
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'

                # new:
                if self.visualize:
                    visualize_r(model_outputs,data)
                    self.visualize = False
                # end new
                
                # calculate pose_metrics
                report_attrs = loss_outputs
                dynamic_counts = {}
                for metric_name in self.pose_metrics:
                    metric_func = POSE_METRICS[metric_name]

                    pred_metric_pose = model_outputs['pred_pose']
                    if 'pred_metric_pose' in model_outputs:
                        pred_metric_pose = model_outputs['pred_metric_pose']

                    # TODO: write write a warning =D

                    future_metric_pose = data['future_pose']
                    if 'future_metric_pose' in data:
                        future_metric_pose = data['future_metric_pose']

                    for action in actions:
                        if action == "all":
                            metric_value = metric_func(pred_metric_pose.to(self.device),
                                                       future_metric_pose.to(self.device),
                                                       self.model.args.keypoint_dim)
                        else:
                            indexes = np.where(np.asarray(data['action']) == action)[0]
                            metric_value = metric_func(pred_metric_pose.to(self.device)[indexes],
                                                       future_metric_pose.to(self.device)[indexes],
                                                       self.model.args.keypoint_dim)
                            dynamic_counts[f'{metric_name}_{action}']=len(indexes)
                        report_attrs[f'{metric_name}_{action}'] = metric_value

                self.reporter.update(report_attrs, batch_size, True, dynamic_counts)

        self.reporter.epoch_finished()
