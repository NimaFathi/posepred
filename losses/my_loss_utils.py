#new (the whole file)

import numpy as np
import matplotlib.pyplot as plt
import torch


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
      
def create_place_holder_dict(device):
    return { "smoking":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "directions":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "discussion":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "eating":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "greeting":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "phoning":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "posing":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],  
                            "purchases":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "sitting":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "sittingdown":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "takingphoto":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "waiting":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "walking":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "walkingdog":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)],
                            "walkingtogether":[0,torch.zeros((25,32)).to( device),-1000*torch.ones((25,32)).to( device),1000*torch.ones((25,32)).to( device)]} #n,sum,max,min
          
    
    

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