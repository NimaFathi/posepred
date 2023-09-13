
import numpy as np
import torch
import torch.nn as nn

#new
import matplotlib.pyplot as plt
#end new

#new:
def visualize_r(y_pred,y_true,sigma,t,action, obs):

    y_pred = y_pred.reshape(25,32,3).clone().detach().cpu().numpy()
    y_true = y_true.reshape(25,32,3).clone().detach().cpu().numpy()
    sigma = sigma.reshape(25,32).clone().detach().cpu().numpy()
    
    fig = plt.figure(figsize=(25,25+10))
    
    fig.suptitle(action, fontsize=32)
    
    Saeeds = [[0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8], [0, 12], [12, 13], [13, 14], [14, 15],[13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27]]
    
    skeleton_old = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    skeleton= [[0,1],[1,2],[2,3],[0,7],[7,8],[8,9],[9,10],[8,14],[14,15],[15,16],  
                      [0,4],[4,5],[5,6],[8,11],[11,12],[12,13] ] #left
    # skeleton_left = [[0,4],[4,5],[5,6],[8,11],[11,12],[12,13]]
    
    KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    
    y_pred = y_pred[:,KeyPoints_from3d,:]
    y_true = y_true[:,KeyPoints_from3d,:]
    sigma = sigma[:,KeyPoints_from3d]
    obs = obs[:,KeyPoints_from3d,:]
    
    for i in range(10):
        ax = fig.add_subplot(5+2, 5, i +1 , projection='3d')
        ydata = obs[i].T[0]/1000
        zdata = obs[i].T[1]/1000
        xdata = obs[i].T[2]/1000
        ax.scatter(xdata,ydata,zdata, color ="grey" , label = "observation" )
        for j in range(16):
            color = "grey" if j<10 else "black"
            ax.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]] , color = color)
        ax.set_title("frame {}".format(-9+i))
        setup_axes(ax)
        
    for i in range(0,25):
        ax = fig.add_subplot(5+2, 5, i +1 +10 , projection='3d')
        
        ydata = y_pred[i].T[0]/1000
        zdata = y_pred[i].T[1]/1000
        xdata = y_pred[i].T[2]/1000
        sigma_ = sigma[i].T
        ax.scatter(xdata,ydata,zdata, color ="mediumvioletred" , label = "prediction" )
        # ax.text(xdata,ydata,zdata, sigma_, color ="mediumvioletred")
        for j in range(16):
            color =  "palevioletred" if i<10 else "pink"
            ax.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]] , color =color)
            if j not in [0,1,4] :
                ax.text(xdata[j], ydata[j], zdata[j], str('%.2f'%sigma_[j]), color = "black")
        
        # for k in range(len(Saeeds)):
        #     ax.plot(xdata[ Saeeds[k]], ydata[Saeeds[k]], zdata[Saeeds[k]] , color = "palevioletred")
        #     ax.text(xdata[k], ydata[k], zdata[k], str('%.2f'%sigma_[k]), color = "black")
        
        ydata = y_true[i].T[0]/1000
        zdata = y_true[i].T[1]/1000
        xdata = y_true[i].T[2]/1000
        
        ax.scatter(xdata,ydata,zdata, color ="turquoise" , label = "ground truth" )
        for j in range(16):
            color = "turquoise" if j<10 else "teal"
            ax.plot(xdata[ skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]] , color = color)
        
        # for k in range(len(Saeeds)):
        #     ax.plot(xdata[ Saeeds[k]], ydata[Saeeds[k]], zdata[Saeeds[k]] , color = "turquoise")
                
        ax.set_title("frame {}".format(i+1 +10 ))
        # ax.view_init(elev=120, azim=-60)
        setup_axes(ax)
           
    fig.tight_layout() 
    plt.legend() 
    plt.savefig(f"./plots/new_vis_{t}.png")
    # plt.close()
    # plt.show()


def setup_axes(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    ax.grid(False)
    ax.set_axis_off()
        
    ax.axes.set_xlim3d(left=-0.45, right=0.45) 
    ax.axes.set_ylim3d(bottom=-0.45, top=0.45) 
    ax.axes.set_zlim3d(bottom=-0.45 , top=0.45 )

#end new



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
            self.t = 0 #new
            
            self.nT = 5
            self.s = self.s.repeat(1, 5, 1)
            self.s[:, :, :] = 0
            self.s[:, 0, :] = init_mean
            self.s[:, 2, :] = 1
            
            
            #new:
            self.sigmas_to_save = []
            self.fig, self.axes = plt.subplots(8, 4, figsize=(16, 10))
            # self.fig.tight_layout()
            #end new
            
            
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
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(5, 32), #new danger used to be 5 for eig values only changed to 5+32 for variance and eig values
                torch.nn.Tanh(), #new dager: used to be ReLU which could be wrong
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(32, 16),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(16, 64),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.2),
                
                torch.nn.Linear(64, 25*32)
            )
                 
            self.t = 0
            #new in new:
            self.sigmas_to_save = []
            
            self.fig, self.axes = plt.subplots(8, 4, figsize=(16, 10))
            self.fig.tight_layout()
            
            #new in new in new
            self.action_u_info = { "smoking":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "directions":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "discussion":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "eating":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "greeting":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "phoning":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "posing":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],  
                                "purchases":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "sitting":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "sittingdown":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "takingphoto":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "waiting":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "walking":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "walkingdog":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)],
                                "walkingtogether":[0,torch.zeros((25,32)).to(self.args.device),-1000*torch.ones((25,32)).to(self.args.device),1000*torch.ones((25,32)).to(self.args.device)]} #n,sum,max,min
            
        elif ('extra_head' in args.time_prior) or (args.time_prior == 'no_u'):    
            self.t = 0
            #new in new:
            self.sigmas_to_save = []
            
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
                batch_size = poses.shape[0]
                eig_value, eig_vectors = self.torch_pca(poses)
                eig_value = eig_value[:,-5:]
                eig_value = eig_value.log() 
                
                            
                #calculate the varience of keypoint positions
                variance = torch.norm(poses.reshape(-1,10,32,3), dim=-1) 
                variance = torch.var(variance, dim=1)
                # eig_value = torch.cat((eig_value, variance), dim=1) #new danger 16, 5+32 / used to be only eig values 16, 5
                
                thetas = self.mlp(eig_value) #B,5
                thetas = thetas.reshape(batch_size, 25, 32)
                
                local_sigma = thetas
                
                
        local_sigma = torch.clamp(local_sigma, min=self.args.clipMinS, max=self.args.clipMaxS)
        
        return local_sigma #local_sigma


    #new:
    def plot_sigmas(self, sigmas, actions):
        
        action_to_color = { "smoking": "pink", "directions":  "coral", "discussion": "slateblue",
                "eating": "magenta","greeting": "tan","phoning": "palevioletred","posing": "teal",  
                "purchases": "navy","sitting": "lime","sittingdown": "green", "takingphoto": "gold",
                "waiting": "gray","walking": "crimson","walkingdog": "crimson","walkingtogether": "crimson"}
        
        b = 0
        for sigma in sigmas: #16,    25, 32
            if "" in actions[b]:
                for i in range(8): 
                    for j in range(4):
                        self.axes[i, j].plot(sigma[:, i*4+j],".", markersize=0.75, color=action_to_color[actions[b]])
                        self.axes[i, j].set_title("joint {}".format(i*4+j))
                        # self.axes[i, j].set_ylim(-1, 8)
            else:
                pass
            b+=1
        
        # markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in action_to_color.values()]
        # plt.legend(markers, action_to_color.keys(), numpoints=1, loc ="upper left", ncol=3 )       
                  
        self.fig.savefig("./plots/new_u.png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        print("saved uncertaity plots .png")
        
        
        
    def plot_sigmas_2(self):
        action_to_color = { "smoking": "pink", "directions":  "coral", "discussion": "slateblue",
                "eating": "magenta","greeting": "tan","phoning": "palevioletred","posing": "teal",  
                "purchases": "navy","sitting": "lime","sittingdown": "green", "takingphoto": "gold",
                "waiting": "gray","walking": "crimson","walkingdog": "crimson","walkingtogether": "crimson"}
        
        fig2 = plt.figure(figsize=(16,10))
        fig2.tight_layout()
        
        for act in self.action_u_info.keys():
            if "pos" in act:
                print(act)
                print(self.action_u_info[act][0])
                for i in range(32):
                    ax2 = fig2.add_subplot(8,4,i+1)
                    # breakpoint()
                    print(action_to_color[act])
                    ax2.plot( ((self.action_u_info[act][1]/self.action_u_info[act][0])[:,i]).detach().cpu().numpy() , "." , color=action_to_color[act] )
                    # ax2.plot(((self.action_u_info[act][2])[:,i]).detach().cpu().numpy() , ":" , color=action_to_color[act] )
                    # ax2.plot(((self.action_u_info[act][3])[:,i]).detach().cpu().numpy() ,":"  , color=action_to_color[act] )
                  
        fig2.savefig("./plots/new_u_maxminmean.png")  
        breakpoint()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        print("_.png")
        plt.close()
    #end new
    
    

    def forward(self, y_pred_, y_true_):
        #new:
        if self.t == 0: print("self.args.time_prior:", self.args.time_prior) 
        if 'extra_head' in self.args.time_prior  :
            sigma = y_pred_['sigmas']
            sigma = sigma.reshape(-1, 25, 32, 3)
            sigma = torch.norm(sigma, dim=-1) #/ 1.73205080757
        elif self.args.time_prior == 'no_u':
            sigma = torch.tensor(0).to(self.args.device) #torch.zeros((y_true_['pred_pose'].shape[0],25,32)).to(self.args.device)
        else: 
            sigma = self.calc_sigma(y_true_) #used to be only this before new
        
        y_pred = y_pred_['pred_pose'] # B,T,JC
        y_true = y_true_['future_pose'] # B,T,JC

        B,T,JC = y_pred.shape
        assert T == self.args.nT and JC % self.args.nJ == 0, "Either number or predicted frames (nT) is not right, or number of joints * dim of each joint is not dividable by nJ"
        J = self.args.nJ
        C = JC // J

        y_pred = y_pred.view(B, T, J, C)
        y_true = y_true.view(B, T, J, C)

        l = torch.norm(y_pred - y_true, dim=-1) # B,T,J
        # l = l/10  #new:
        l = torch.mean(torch.exp(-sigma) * l + sigma)
        
        #new
        # if self.args.time_prior == 'r_m':
        #     for i in range(B):
        #         act = y_true_['action'][i]
        #         self.action_u_info[act][0] += 1
        #         self.action_u_info[act][1] += sigma[i]
        #         self.action_u_info[act][2] = torch.max(self.action_u_info[act][2], sigma[i])
        #         self.action_u_info[act][3] = torch.min(self.action_u_info[act][3], sigma[i])
        
        #new:
        if self.t<0:
            self.plot_sigmas(sigma.detach().cpu().numpy(), y_true_["action"])
               
            meow = torch.argmin(torch.abs(sigma))
            meow_J = meow % 32
            meow_T = (meow //32) % 25
            meow_B = ((meow //32)//25)%16
            meow = sigma[meow_B,meow_T]
            visualize_r(y_pred[meow_B], y_true[meow_B], sigma[meow_B], self.t , y_true_['action'][meow_B],
                        y_true_['observed_pose'][meow_B].detach().cpu().numpy().reshape(50, J, C))
            
            
        elif self.t==10:
            plt.close(self.fig)
            
        # elif self.t == 200:
        #     self.plot_sigmas_2()
 
        self.t+=1
        #end new
        
        return {
          'loss' : l
        }
        

        #new:
        # sigma = sigma * (1/(1+np.exp(-1*self.t/200)))
        #end new
