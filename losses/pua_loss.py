
import numpy as np
import torch
import torch.nn as nn

#new
import matplotlib.pyplot as plt
from models.st_transformer.data_proc import Human36m_Preprocess
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
                ax.text(xdata[j], ydata[j], zdata[j], str('%.2f'%sigma_[j]), color = "black", fontsize=18)
        
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
    plt.savefig(f"./plots/new_vis_{t}.pdf")
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


def visualize_pca(action, eig_value_):
    eig_value = eig_value_.clone().detach().cpu().numpy()
    eig_value[eig_value < 1e-4] = 0
    eig_value = np.log(eig_value)
    fig1 = plt.figure(figsize=(10,10))
    fig2 = plt.figure(figsize=(20,10))
    
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    
    action_to_color = { "smoking": "pink", "directions":  "coral", "discussion": "slateblue",
                "eating": "magenta","greeting": "tan","phoning": "palevioletred","posing": "teal",  
                "purchases": "navy","sitting": "lime","sittingdown": "green", "takingphoto": "gold",
                "waiting": "gray","walking": "crimson","walkingdog": "crimson","walkingtogether": "crimson"}
    
    action_count = { "smoking": 0, "directions":  0, "discussion": 0,
                "eating": 0,"greeting": 0,"phoning": 0,"posing": 0,  
                "purchases": 0,"sitting": 0,"sittingdown": 0, "takingphoto": 0,
                "waiting": 0,"walking": 0,"walkingdog": 0,"walkingtogether": 0}
    
    markers = ["v", "o",  "^","*", "s", "<", ">",  "p", "P", "h", "H", "+", "x", "X", "D", "d"]
    
    #plot the eigen values for each first dimention 
    for i in range(eig_value.shape[0]):
        action_label = action[i]
        action_count[action_label] += 1
        ax1.plot([i] * eig_value.shape[1], eig_value[i], color = "gray", marker = "o", linestyle='')
        # ax1.plot([i] * 5, eig_value[i, -5: ], label = f"eig {i}",  marker = ".", color = "mediumvioletred", linestyle='')
        ax1.plot([i] * 5, eig_value[i, -5: ],  marker = ".", color = action_to_color[action_label], linestyle='')
        
        #marker = markers[action_count[action_label]] 
        ax2.plot([action_label] * 5, eig_value[i, -5: ], label = f"eig {i}",  marker = "o", linestyle='', markersize=10)
    
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in action_to_color.values()]
    fig1.legend(markers, action_to_color.keys())   #, loc ="upper left" , numpoints=1, ncol=3
        
    # fig1.legend()    
    fig1.tight_layout() 
    fig2.tight_layout() 
    fig1.savefig(f"./plots/eig_values.pdf")
    fig2.savefig(f"./plots/eig_values_2_.pdf")
    
    
    # breakpoint()

        

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
        
        #new:
        self.t = 0 
        
        self.sigmas_to_save = []
        self.fig, self.axes = plt.subplots(8, 4, figsize=(16, 10))
        self.fig.tight_layout()
        
        self.preprocess = Human36m_Preprocess(args).to(args.device)
        
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
          
        #end new
        
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
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(5, 32), #new danger used to be 5 for eig values only changed to 5+32 for variance and eig values
                torch.nn.Tanh(), #new dager: used to be ReLU which could be wrong
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(32, 16),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(16, 64),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(64, 25*32)
            )
            
        elif args.time_prior == 'all_mlp':
            self.action_list = args.action_list
            self.nA = len(self.action_list)
            self.action_map = {self.action_list[i]: i for i in range(self.nA)}

            self.embed_layer_action = nn.Linear(16, 16) #1,37
            self.nT = 5
            self.s = self.s.repeat(1, 5, 1)
            self.s[:, :, :] = 0
            self.s[:, 0, :] = init_mean
            self.s[:, 2, :] = 1
            
            self.embed_layer_1 = nn.Linear(5, 16)
            self.embed_layer_2 = nn.Linear(32, 32)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(16+16+32, 128), #new danger used to be 5 for eig values only changed to 5+32 for variance and eig values 5+32
                torch.nn.Tanh(), #new dager: used to be ReLU which could be wrong
                torch.nn.Dropout(0.4),
                
                torch.nn.Linear(128, 64),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.4),
                
                torch.nn.Linear(64, 128),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.4),
                
                torch.nn.Linear(128, 512),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.4),
                
                torch.nn.Linear(512, 25*32) #800(25*32) 25*22(550) 25*29(725)
            )
            
            
        elif args.time_prior == 'theta_mlp':
            self.action_list = args.action_list
            self.nA = len(self.action_list)
            self.action_map = {self.action_list[i]: i for i in range(self.nA)}
            
            # self.embed_layer_action = nn.Linear(16, 16) #1,37
            
            self.nT = 5
            self.s = self.s.repeat(1, 5, 1)
            self.s[:, :, :] = 0
            self.s[:, 0, :] = init_mean
            self.s[:, 2, :] = 1
            
            self.embed_layer_1 = nn.Linear(5, 5)
            self.embed_layer_2 = nn.Linear(32, 32)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(16, 64), #new danger used to be 5 for eig values only changed to 5+32 for variance and eig values 5+32
                torch.nn.Tanh(), #new dager: used to be ReLU which could be wrong
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(64, 32),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(32, 128),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(128, 512),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(512, 25*32) #800(25*32) 25*22(550) 25*29(725)
            )
            # breakpoint()
            # self.mlp = torch.nn.Sequential(
            #     torch.nn.Linear(37, 64), #new danger used to be 5 for eig values only changed to 5+32 for variance and eig values
            #     torch.nn.Tanh(), #new dager: used to be ReLU which could be wrong
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(64, 32),
            #     torch.nn.Tanh(),
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(32, 128),
            #     torch.nn.Tanh(),
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(128, 512),
            #     torch.nn.Tanh(),
            #     torch.nn.Dropout(0.3),
                
            #     torch.nn.Linear(512, 25*32)
            # )
            
                
 
        elif ('extra_head' in args.time_prior) or (args.time_prior == 'no_u') or  ('_' in args.time_prior):    
            pass
            # self.t = 0
            # #new in new:
            # self.sigmas_to_save = []
            
            # self.fig, self.axes = plt.subplots(8, 4, figsize=(16, 10))
            # self.fig.tight_layout()
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
                poses = self.preprocess(poses) #new in new !!!
                batch_size = poses.shape[0]
                n_o = poses.shape[1]
                
                eig_value, eig_vectors = self.torch_pca(poses)
                if self.t == 0:
                    visualize_pca(y_true['action'], eig_value)
                eig_value = eig_value[:,-5:]
                # print("eig_value",eig_value)
                # eig_value = eig_value.log() #new DANGERRRRRRRRRRRRRRRRRRRRRR
                # print("log eig_value",eig_value) 
                
                            
                #calculate the varience of keypoint positions
                # variance = torch.norm(poses.reshape(-1,n_o,32,3), dim=-1) 
                # variance = torch.var(variance, dim=1)
                # eig_value = torch.cat((eig_value, variance), dim=1) #new danger 16, 5+32 / used to be only eig values 16, 5
                
                thetas = self.mlp(eig_value) 
                thetas = thetas.reshape(batch_size, 25, 32)
                
                local_sigma = thetas
                
                # breakpoint()
                data_tensor = poses.view(poses.shape[0], -1, poses.shape[-1])

                # Compute the FFT along the second dimension (time dimension)
                fft_result_tensor = torch.fft.fft(data_tensor, dim=1)
                fft_freqs_tensor = torch.fft.fftfreq(data_tensor.shape[1])

                # Calculate the magnitude of the FFT
                fft_magnitude_tensor = torch.abs(fft_result_tensor)

                # Plot the magnitude of the FFT for each sample in the batch
                plt.close('all')
                markers = ['D', 'v', 'o']
                for b in range(data_tensor.shape[0]):
                    plt.figure(figsize=(10, 6))
                    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
                    for i in range(int(fft_magnitude_tensor.shape[2]/3)): #range(fft_magnitude_tensor.shape[2]):
                        # plt.plot(fft_freqs_tensor, fft_magnitude_tensor[b, :, i], label=f'Joint {i}')
                        #detach and make numpy array before ploting:
                        # plt.plot(fft_freqs_tensor.detach().cpu().numpy(), fft_magnitude_tensor[b, :, i].detach().cpu().numpy(), label=f'Joint {i//3}', marker=markers[i%3], linestyle = 'None', markerfacecolor='None')
                        axs[0].plot(fft_freqs_tensor.detach().cpu().numpy(), fft_magnitude_tensor[b, :, i].detach().cpu().numpy(), label=f'Joint {i}', marker=markers[0], linestyle = 'None', markerfacecolor='None')
                        axs[1].plot(fft_freqs_tensor.detach().cpu().numpy(), fft_magnitude_tensor[b, :, i+1].detach().cpu().numpy(), label=f'Joint {i}', marker=markers[1], linestyle = 'None', markerfacecolor='None')
                        axs[2].plot(fft_freqs_tensor.detach().cpu().numpy(), fft_magnitude_tensor[b, :, i+2].detach().cpu().numpy(), label=f'Joint {i}', marker=markers[2], linestyle = 'None', markerfacecolor='None')


                    plt.xlabel('Frequency')
                    plt.ylabel('Magnitude')
                    # plt.title('FFT Magnitude for Joint Movement Data - ' +y_true['action'][b]+ f' {b}')
                    #title on top of the whole image:
                    plt.suptitle('FFT Magnitude for Joint Movement Data - ' +y_true['action'][b]+ f' {b}')
                    # plt.legend()
                    #seperate legend for each subplot:
                    axs[0].legend()
                    axs[1].legend()
                    axs[2].legend()
                    plt.show()
                    
                    #saving the image:
                    plt.savefig(f"./plots/fft_{b}.png")
                    plt.close()
                    breakpoint()
                
            elif self.args.time_prior == 'mlp_sig5':
                poses = y_true['observed_pose']
                
            elif self.args.time_prior == 'theta_mlp': #only action mlp
                actions = y_true['action']
                indx = torch.tensor([self.action_map[act] for act in actions]).to(self.args.device)
                temp = torch.nn.functional.one_hot(indx, num_classes=16)
                temp = temp.float()
                thetas = self.mlp(temp.float())
                thetas = thetas.reshape(batch_size, 25, 32)
                thetas[:,:,[0,1,6,11]] = -1
                local_sigma = thetas
            
            elif self.args.time_prior == 'all_mlp':
                
                actions = y_true['action']
                indx = torch.tensor([self.action_map[act] for act in actions]).to(self.args.device)
                
                temp = torch.nn.functional.one_hot(indx, num_classes=16)
                temp = temp.float()
                action_embeded = self.embed_layer_action(temp.float())
                
                
                poses = y_true['observed_pose']
                batch_size = poses.shape[0]
                n_o = poses.shape[1]
                
                eig_value, eig_vectors = self.torch_pca(poses)
                eig_value = eig_value[:,-5:]
                eig_value = eig_value.log() 
                eig_value_embeded = self.embed_layer_1(eig_value)
                

                variance = torch.norm(poses.reshape(-1,n_o,32,3), dim=-1) 
                variance = torch.var(variance, dim=1)
                variance_embeded = self.embed_layer_2(variance)
                
                mlp_inp = torch.cat((eig_value_embeded, variance_embeded, action_embeded), dim=1) #5+32
                # print(mlp_inp.shape)
                # breakpoint()
                try:
                    # thetas = self.mlp(mlp_inp) #B,5
                    thetas = self.mlp(mlp_inp.float()) #B,5
                except:
                    print("TRY EXCEPT ERROR")
                    breakpoint()
                
                thetas = thetas.reshape(batch_size, 25, 32)
                thetas[:,:,[0,1,6,11]] = -1
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
                        self.axes[i, j].plot(sigma[:, i*4+j],".", markersize=3, color=action_to_color[actions[b]])
                        self.axes[i, j].set_title("joint {}".format(i*4+j))
                        # self.axes[i, j].set_ylim(-1, 8)
            else:
                pass
            b+=1
        
        # markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in action_to_color.values()]
        # plt.legend(markers, action_to_color.keys(), numpoints=1, loc ="upper left", ncol=3 )       
        
        self.fig.savefig("./plots/Uncertaties.png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        print("saved uncertaity plots .pdf")
        
        
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
        # breakpoint()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        print("_.png")
        plt.close()
    #end new
    
        

    def test_(self, y_true_, name):
        skeleton= [[0,1],[1,2],[2,3],[0,7],[7,8],[8,9],[9,10],[8,14],[14,15],[15,16],  
                      [0,4],[4,5],[5,6],[8,11],[11,12],[12,13] ] #left

        KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    
        y_true = y_true_.detach().cpu().numpy()
        
        B,T,JC = y_true.shape
        y_true = y_true.reshape(B,T,32,3)
        y_true = y_true[:,:,KeyPoints_from3d,:]
        
        link_lenghths = np.zeros((B,T,16))
          
        for i in range(16):
            l = y_true[:,:,skeleton[i][0],:] - y_true[:,:,skeleton[i][1],:] #(B,T,1,3)
            l2 = (l[:,:,0]**2 + l[:,:,1]**2 + l[:,:,2]**2)**0.5 #(B,T,1
            link_lenghths[:,:,i] = l2/10
            # link_lenghths[:,:,i] = np.linalg.norm(l, axis=-1)/100
            
                
        fig3 = plt.figure(figsize=(16,10))
        fig3.tight_layout()
        
        for i in range(16):
            ax2 = fig3.add_subplot(4,4,i+1) #for each link
            ax2.set_title("link {}".format(i))
            for j in range(B):
                ax2.plot(link_lenghths[j,:,i])
                    
        fig3.savefig("./plots/test_"+name+".png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        print("HELP")
        plt.close()
    
        
    def test_sk(self, y_true_, name):
        skeleton= [[0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8], [0, 12], [12, 13], 
                   [13, 14], [14, 15],[13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27]] #saeed's

        KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    
        y_true = y_true_.detach().cpu().numpy()
        
        B,T,JC = y_true.shape
        y_true = y_true.reshape(B,T,32,3)
        # y_true = y_true[:,:,KeyPoints_from3d,:]
        
        link_lenghths = np.zeros((B,T,16))
          
        for i in range(16):
            l = y_true[:,:,skeleton[i][0],:] - y_true[:,:,skeleton[i][1],:] #(B,T,1,3)
            l2 = (l[:,:,0]**2 + l[:,:,1]**2 + l[:,:,2]**2)**0.5 #(B,T,1
            link_lenghths[:,:,i] = l2/10
            # link_lenghths[:,:,i] = np.linalg.norm(l, axis=-1)/100
            
                
        fig3 = plt.figure(figsize=(16,10))
        fig3.tight_layout()
        
        for i in range(16):
            ax2 = fig3.add_subplot(4,4,i+1) #for each link
            ax2.set_title("link {}".format(i))
            for j in range(B):
                ax2.plot(link_lenghths[j,:,i])
                    
        fig3.savefig("./plots/test_"+name+".png")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        print("HELP")
        plt.close()
        
        
    def cal_link_len_error(self, y_true, y_pred):
        skeleton= [[0,1],[1,2],[2,3],[0,7],[7,8],[8,9],[9,10],[8,14],[14,15],[15,16],  
                      [0,4],[4,5],[5,6],[8,11],[11,12],[12,13] ] #left
        
        KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        y_true = y_true[:,:,KeyPoints_from3d,:]
        y_pred = y_pred[:,:,KeyPoints_from3d,:]
        
        input_len = torch.zeros((y_true.shape[0], 16)).to(self.args.device)
        output_lens = torch.zeros((y_pred.shape[0], y_pred.shape[1],16)).to(self.args.device)
        for i in range(16):
            # breakpoint()
            input_len[:,i] = torch.mean(torch.norm(y_true[:,:,skeleton[i][0],:] - y_true[:,:,skeleton[i][1],:], dim=-1), dim=1)
            output_lens[:,:,i] = torch.norm(y_pred[:,:,skeleton[i][0],:] - y_pred[:,:,skeleton[i][1],:], dim=-1)
            
        len_error = (torch.abs( output_lens - input_len.unsqueeze(1)))
        len_error = torch.mean(len_error)
        
        return len_error #B,T,16
            

    def forward(self, y_pred_, y_true_):
        
        
        # breakpoint()
        plt.close()
        y_pred = y_pred_['pred_pose'] # B,T,JC
        y_true = y_true_['future_pose']
        observed_data = y_true_['observed_pose']
        test_data_ = torch.cat((observed_data, y_pred), dim=1)
        test_data_ = self.preprocess(test_data_)
        minn = test_data_.min().cpu().detach().numpy()
        maxx = test_data_.max().cpu().detach().numpy()

        for t_ in range(min(16, y_true.shape[0])):
            test_data = test_data_[t_]
            test_data = test_data.cpu().detach().numpy()
            test_data = test_data.reshape(75, -1 , 3)
            test_data = (test_data-minn)/(maxx-minn)
            temp = test_data[50,:,:]
            test_data_rl = test_data - temp + 0.5
            temp = test_data[0,:,:]
            test_data_rf = test_data - temp + 0.5
            
            #creating 3 subplots with 3 images of test_data and test_data_rf and saving the image:
            fig, axs = plt.subplots(1, 3, figsize=(10, 10))
            axs[0].imshow(test_data)
            axs[1].imshow(test_data_rf)
            axs[2].imshow(test_data_rl)
            #add action as the title to figure:
            fig.suptitle(y_true_['action'][t_])
               
            plt.savefig('./plots/image'+str(t_)+'.png')
        breakpoint()
        
        
        
        # if self.t==0:
        #     self.test_(y_true_["observed_pose"],"observed_pose")
        #     self.test_(y_pred_["pred_pose"],"pred_pose")
        #     self.test_(torch.cat((y_true_["observed_pose"], y_pred_["pred_pose"]), dim=1),  "obs_pred_pose")
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
        # l = l/10 #new:
        # print(l, self.cal_link_len_error(y_true, y_pred) )
        # l = self.cal_link_len_error(y_true, y_pred)
        
        
        #new
        # if self.args.time_prior == 'r_m':
        #     for i in range(B):
        #         act = y_true_['action'][i]
        #         self.action_u_info[act][0] += 1
        #         self.action_u_info[act][1] += sigma[i]
        #         self.action_u_info[act][2] = torch.max(self.action_u_info[act][2], sigma[i])
        #         self.action_u_info[act][3] = torch.min(self.action_u_info[act][3], sigma[i])
        
        #new:
        if (self.t<10) and (self.args.time_prior != 'no_u') :
            self.plot_sigmas(sigma.detach().cpu().numpy(), y_true_["action"])
               
            meow = torch.argmin(torch.abs(sigma))
            meow_J = meow % 32
            meow_T = (meow //32) % 25
            meow_B = ((meow //32)//25)%16
            meow = sigma[meow_B,meow_T]
            
            meow_B = 0

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



def inverse_fft(fft_amp, fft_pha):
    imag = fft_amp * torch.sin(fft_pha)
    real = fft_amp * torch.cos(fft_pha)
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifft(fft_y)
    return y 