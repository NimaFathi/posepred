
import numpy as np
import torch
import torch.nn as nn

class PUALoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        '''
        args mush have:
        @param init_mean : float : initialize S such that the mean of S is init_mean, 3.5 is a good default
        @param tasks : str : list of tasks as a string, J for tasks over joints, T for over time(frames), A for actions. if A is used, 'action' must be in the input.
        @param nT: int: number of frames to predict
        @param nJ: int: number of joints
        @param action_list : list(str) : name of different actions as a list of str. used in case of A present in tasks.
        @param time_prior: str : what time prior to use, must be one of sig5, sig*, none
        @param clipMinS, clipMaxS: float : these values are used to slip s. MinS is needed if there are tasks in the input with errors near zero. one can set them to None, resulting in no cliping.
        @param device : str : device to run torch on
        '''

        self.args = args
        init_mean = self.args.init_mean
        self.s = torch.ones(1, 1, 1, requires_grad = True).to(self.args.device) * init_mean
        # Fix tasks for joints
        if 'J' in args.tasks:
            self.nJ = args.nJ
            self.s = self.s.repeat(1, 1, self.nJ)
        else:
            self.nJ = 1
        #fix tasks for time
        if 'T' not in args.tasks:
            self.nT = 1
        elif args.time_prior == 'sig5':
            self.nT = 5
            self.s = self.s.repeat(1, 5, 1)
            self.s[:, :, :] = 0
            self.s[:, 0, :] = init_mean
            self.s[:, 2, :] = 1
        elif args.time_prior == 'sig*':
            self.nT = 3
            self.s = self.s.repeat(1, 3, 1)
            self.s[:, 0, :] = init_mean
            self.s[:, 1, :] = 1
            self.s[:, 2, :] = -10
        elif args.time_prior == 'sig2':
            self.nT = 2
            self.s = self.s.repeat(1, 2, 1)
            self.s[:, 0, :] = 1
            self.s[:, 1, :] = -10
        elif args.time_prior == 'log*':
            self.nT = 3
            self.s = self.s.repeat(1, 3, 1)
            self.s[:, 0, :] = 1
            self.s[:, 1, :] = 1
            self.s[:, 2, :] = init_mean
        elif args.time_prior == 'none':
            self.nT = args.nT
            self.s = self.s.repeat(1, self.nT, 1)
        elif 'poly' in args.time_prior:
            self.nT = int(args.time_prior[4:]) + 1
            self.s = self.s.repeat(1, self.nT, 1)
            self.s[:, 1:, :] = 0
        #new:
        elif args.time_prior == 'r_m':
            self.nT = 5
            self.s = self.s.repeat(1, 5, 1)
            self.s[:, :, :] = 0
            self.s[:, 0, :] = init_mean
            self.s[:, 2, :] = 1
            
            # self.mlp = torch.nn.Sequential(
            #     torch.nn.Linear(16, 64),
            #     torch.nn.ReLU(),
            #     torch.nn.Dropout(0.3),
                 
            #     torch.nn.Linear(64, 256),
            #     torch.nn.ReLU(), 
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(256, 1024), 
            #     torch.nn.ReLU(),
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(1024, 800),
            #     torch.nn.Sigmoid())
            
            # self.mlp = torch.nn.Sequential(
            #     torch.nn.Linear(16, 64),
            #     torch.nn.ReLU(),
            #     torch.nn.Dropout(0.3),
                 
            #     torch.nn.Linear(64, 256),
            #     torch.nn.ReLU(), 
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(256, 1024), 
            #     torch.nn.ReLU(),
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(1024, 800),
            #     torch.nn.Sigmoid())
            
            # self.mlp2 = torch.nn.Sequential(
            #     torch.nn.Linear(5, 64),
            #     torch.nn.ReLU(),
            #     torch.nn.Dropout(0.3),
                 
            #     torch.nn.Linear(64, 128),
            #     torch.nn.ReLU(), 
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(128, 512), 
            #     torch.nn.ReLU(),
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(512, 160))
            
            self.count = 0
            
            #new in new:
            self.sigmas_to_save = []
            import matplotlib.pyplot as plt
            self.fig, self.axes = plt.subplots(8, 4, figsize=(16, 10))
            self.fig.tight_layout()
            
            
        #end new
        else:
            raise Exception("{} is not a supported prior for time axis, options are: [sig5, sig*, none].".format(args.time_prior))
        # fix tasks for action
        if 'A' in args.tasks:
            self.action_list = args.action_list
            self.nA = len(self.action_list)
            self.action_map = {self.action_list[i]: i for i in range(self.nA)}
            self.s = self.s.repeat(self.nA, 1, 1)
            self.sigma = nn.Embedding(self.nA, self.nT * self.nJ)
            self.sigma.weight = nn.Parameter(self.s.view(-1, self.nT * self.nJ))
        else:
            self.nA = None
            self.sigma = nn.Parameter(self.s)
            #new:
            # self.sig5_pose_theta = nn.Parameter(torch.ones(1, 5, 1))
            #new danger:
            # self.sig5_pose_theta = nn.Parameter(torch.ones(16, 1, 32))
            
    #new:
    def torch_pca(self, X):
        centred_X = X - X.mean(dim=1, keepdim=True)
        covariance_X = (centred_X.transpose(1,2) @ centred_X) / (X.shape[1] - 1)
        eig_value, eig_vectors = torch.linalg.eigh(covariance_X)
        return eig_value, eig_vectors
    #end new

    def calc_sigma(self, y_true):
        local_sigma = self.sigma
        if self.nA is not None:
            actions = y_true['action']
            indx = torch.tensor([self.action_map[act] for act in actions]).to(self.args.device)
            local_sigma = local_sigma(indx)
            local_sigma = local_sigma.view(-1, self.nT, self.nJ)
        
        if 'T' in self.args.tasks:
            if self.args.time_prior == 'sig5':
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(0) # 1, T, 1
                c = 2 * local_sigma[:, 3 - 1, :] * local_sigma[:, 5 - 1, :] / torch.abs(local_sigma[:, 3 - 1, :] + local_sigma[:, 5 - 1, :])
                f = 1 / (1 + torch.exp(-c * (local_sigma[:, 4 - 1, :] - x)))
                g = torch.exp(local_sigma[:, 3 - 1, :] * (local_sigma[:, 4 - 1, :] - x))
                h = torch.exp(local_sigma[:, 5 - 1, :] * (local_sigma[:, 4 - 1, :] - x))
                local_sigma = local_sigma[:, 1 - 1, :] + (local_sigma[:, 2 - 1, :] / (1 + f * g + (1 - f) * h))
                
            elif self.args.time_prior == 'sig*':
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(0) # 1, T, 1
                local_sigma = local_sigma[:, 0:1, :] / (1 + torch.exp(local_sigma[:, 1:2, :] * (local_sigma[:, 2:3, :] - x)))
            elif self.args.time_prior == 'sig2':
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(0) # 1, T, 1
                local_sigma = self.args.init_mean / (1 + torch.exp(local_sigma[:, 0:1, :] * (local_sigma[:, 1:2, :] - x)))
            elif self.args.time_prior == 'log*':
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(0) # 1, T, 1
                local_sigma = local_sigma[:, 0:1, :] / torch.log(x + torch.abs(local_sigma[:, 1:2, :]) + 1e-4) + local_sigma[:, 2:3, :]
            elif 'poly' in self.args.time_prior:
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(1).unsqueeze(0) / 10 # 1, T, 1, 1
                po = torch.arange(self.nT).to(self.args.device).unsqueeze(1).unsqueeze(0).unsqueeze(0) # 1, 1, D, 1
                x = x ** po # 1, T, D, 1
                local_sigma = local_sigma.unsqueeze(1) # 1, 1, D, ?
                local_sigma = (local_sigma * x).sum(dim=-2) # 1, T, ?
                
            #new:
            elif self.args.time_prior == 'r_m':
                poses = y_true['observed_pose']
                eig_value, eig_vectors = self.torch_pca(poses)
                # eig_value = eig_value[:,-16:]
                
                # temp = self.mlp(eig_value)
                # temp = temp.reshape(-1, 25, 32)
                # self.count += 1
                # if self.count > 10000:
                #     print("eig_value[0][0]",eig_value[0][0])
                #     print("temp[0][0]",temp[0][0])
                #     self.count = 0

                # eig_value = eig_value[:,-5:]
                # temp = self.mlp2(eig_value/300000)
                # temp = temp.reshape(-1, 5, 32)
                # local_sigma = temp # 16 , 5 , 32
                # breakpoint()
                
                # sig5_pose_theta = self.sig5_pose_theta
                # poses = poses.reshape(poses.shape[0], poses.shape[1], -1, 3) #B, T, 32, 3
                # x = torch.norm(poses, dim=-1) 
                # x = torch.max(x, dim=-1) - torch.min(x, dim=-1)
                # x = (x[:,-1:,:]-x[:,0:1,:]).abs() + 0.25 #average abs_velocity + 0.25 to avoid zeros
                # breakpoint()
                # c = 2 * sig5_pose_theta[:, 2:3, :] * sig5_pose_theta[:, 4:5, :] / torch.abs(sig5_pose_theta[:, 2:3, :] + sig5_pose_theta[:, 4:5, :])
                # f = 1 / (1 + torch.exp(-c * (sig5_pose_theta[:, 3:4, :] - x)))
                # g = torch.exp(sig5_pose_theta[:, 2:3, :] * (sig5_pose_theta[:, 3:4, :] - x))
                # h = torch.exp(sig5_pose_theta[:, 4:5, :] * (sig5_pose_theta[:, 3:4, :] - x))
                # sig5_pose = sig5_pose_theta[:, 0:1, :] + (sig5_pose_theta[:, 1:2, :] / (1 + f * g + (1 - f) * h))
                
                x = torch.arange(self.args.nT).to(self.args.device).unsqueeze(1).unsqueeze(0) # 1, T, 1
                c = 2 * local_sigma[:, 3 - 1, :] * local_sigma[:, 5 - 1, :] / torch.abs(local_sigma[:, 3 - 1, :] + local_sigma[:, 5 - 1, :])
                f = 1 / (1 + torch.exp(-c * (local_sigma[:, 4 - 1, :] - x)))
                g = torch.exp(local_sigma[:, 3 - 1, :] * (local_sigma[:, 4 - 1, :] - x))
                h = torch.exp(local_sigma[:, 5 - 1, :] * (local_sigma[:, 4 - 1, :] - x))
                local_sigma = local_sigma[:, 1 - 1, :] + (local_sigma[:, 2 - 1, :] / (1 + f * g + (1 - f) * h))
                
                
                #new:
                # local_sigma = (sig5_pose + local_sigma.repeat())/2
             
            #end new
        
        local_sigma = torch.clamp(local_sigma, min=self.args.clipMinS, max=self.args.clipMaxS)
        return local_sigma


    #new:
    def plot_sigmas(self, sigmas):
        
        for sigma in sigmas: #16,    25, 32
            for i in range(8): 
                for j in range(4):
                    self.axes[i, j].plot(sigma[:, i*4+j],".")
                    # breakpoint()
                    self.axes[i, j].set_title("joint {}".format(i*4+j))
                    # breakpoint()
                    # self.axes[i, j].set_ylim(-1, 2)
                    
        self.fig.savefig("/home/rh/codes/posepred/my_temp/sigmas.png")
        print("saved sigmas.png")
    #end new

    def forward(self, y_pred_, y_true):
        
        #new:
        if self.args.time_prior == 'r_m':
            # breakpoint()
            sigma = y_pred_['sigmas']
            sigma = sigma.reshape(-1, 25, 32, 3)
            sigma = torch.norm(sigma, dim=-1) / 1.73205080757
        else:
            sigma = self.calc_sigma(y_true)
        
        # breakpoint()

        y_pred = y_pred_['pred_pose'] # B,T,JC
        y_true = y_true['future_pose'] # B,T,JC

        B,T,JC = y_pred.shape
        assert T == self.args.nT and JC % self.args.nJ == 0, "Either number or predicted frames (nT) is not right, or number of joints * dim of each joint is not dividable by nJ"
        J = self.args.nJ
        C = JC // J

        y_pred = y_pred.view(B, T, J, C)
        y_true = y_true.view(B, T, J, C)

        l = torch.norm(y_pred - y_true, dim=-1) # B,T,J
        l = torch.mean(torch.exp(-sigma) * l + sigma)
        
        t=0
        if t<0:
            self.plot_sigmas(sigma.detach().cpu().numpy())
            t+=1
            # breakpoint()
            
        
        #does it make sense to do this: #new danger:
        #l0 = torch.norm(y_pred - y_true, dim=-1) # B,T,J
        #l1 = torch.exp(-sigma1) * l0 + sigma1 # B,T,J
        #l2 = torch.mean(l1, dim=0) # B,J
        #l3 = torch.exp(-sigma2) * l2 + sigma2 #sigma2: spatial # B,J
        #l = torch.mean(l2) 

        return {
          'loss' : l
        }