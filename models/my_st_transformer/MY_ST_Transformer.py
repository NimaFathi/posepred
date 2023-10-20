import math

import torch
import torch.nn.functional as F
from torch import nn

from models.st_transformer.data_proc import Preprocess, Postprocess, Human36m_Postprocess, Human36m_Preprocess, AMASS_3DPW_Postprocess, AMASS_3DPW_Preprocess


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

#new:
def Conv1d_with_init_padding(in_channels, out_channels, kernel_size, padding):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    nn.init.kaiming_normal_(layer.weight)
    return layer
#new
def Conv2d_(in_channels, out_channels, kernel_size):
    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer
#nd new


class diff_CSDI(nn.Module):
    def __init__(self, args, inputdim, side_dim):
        super().__init__()
        self.args = args
        self.channels = args.diff_channels

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    nheads=args.diff_nheads
                )
                for _ in range(args.diff_layers)
            ]
        )
        
        #new
        self.mod = self.args['loss']['time_prior'][11:] #extra_head_(A,B,C,...)
        
        if "B" in self.mod or "A" in self.mod:
            self.output_projection_sigma_0 = Conv1d_with_init(66, 66, 1)
        
        if "A" in self.mod:
            
            self.output_projection_sigma_1 = Conv1d_with_init(66, 66, 1)
            self.output_projection_sigma_2 = Conv1d_with_init(66, 66, 1)  
            
            self.output_projection_sigma_second = Conv1d_with_init(66*3, 66, 1) #nn 6 3
            self.output_projection_sigma_third = Conv1d_with_init(66, 66, 1)   
        
            if self.mod == "A6" :
                self.output_projection_sigma_00 = Conv1d_with_init(66, 66, 1) #nn
                self.output_projection_sigma_11 = Conv1d_with_init(66, 66, 1) #nn
                self.output_projection_sigma_22 = Conv1d_with_init(66, 66, 1) #nn   
                self.output_projection_sigma_first = Conv1d_with_init_padding(66*6, 66*3, 3, 1) #nn 6 3  
            else:
                self.output_projection_sigma_first = Conv1d_with_init_padding(66*3, 66*3, 3, 1)

        elif self.mod == "B":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(66*self.args["obs_frames_num"], 1024*2),  #4950
                torch.nn.Tanh(),
                torch.nn.Dropout(0.4),
                
                torch.nn.Linear(1024*2, 1024), #new in NEW (used to be 1024 to 256 from first)
                torch.nn.Tanh(), #used to be ReLU before new in new
                torch.nn.Dropout(0.3), #used to be 0.2 in frist NEW
                
                torch.nn.Linear(1024, 256),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(256, 64),
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(64, 256), 
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(256, 512), #new in NEW (used to be 64 to 256 to the last )
                torch.nn.Tanh(),
                torch.nn.Dropout(0.3),
                
                torch.nn.Linear(512, 66*self.args["pred_frames_num"])
            )
            
 
    def sigma_network_B(self, skip_sigma):
        x0 = self.output_projection_sigma_0(skip_sigma) #66 75
        x0 = x0[...,:self.args["obs_frames_num"]] #B, 66, obs(10 or 50)
        F.relu(x0)
        x0 = x0.reshape(-1, 66*self.args["obs_frames_num"])
        x0 = self.mlp(x0)
        # zeros = torch.zeros(x0.size(0), 66*self.args["obs_frames_num"], device=x0.device)
        # x0 = torch.cat([zeros, x0], dim=1)
        x0 = x0.view(-1, 66, self.args["pred_frames_num"] )
        return x0    
    
    def sigma_network_A_conv(self, skip_sigma): #new
        # breakpoint()
        x0 = self.output_projection_sigma_0(skip_sigma[0])
        F.relu(x0)
        x1 = self.output_projection_sigma_1(skip_sigma[1])
        F.relu(x1)
        x2 = self.output_projection_sigma_2(skip_sigma[2])
        F.relu(x2)
        
        if self.mod == "A6" :
            x00 = self.output_projection_sigma_00(skip_sigma[3]) #nn
            F.relu(x0)
            x11 = self.output_projection_sigma_11(skip_sigma[4]) #nn
            F.relu(x1)
            x22 = self.output_projection_sigma_22(skip_sigma[5]) #nn
            F.relu(x2)        
        
        
        if self.mod == "A6" :
            x_ = torch.cat((x0, x1, x2, x00,x11,x22), dim=1)
        else:
            x_ = torch.cat((x0, x1, x2), dim=1) #nn 3 6


        x_ = self.output_projection_sigma_first(x_) #new in new
        F.relu(x_)
        x_ = self.output_projection_sigma_second(x_)
        F.relu(x_)
        x_ = self.output_projection_sigma_third(x_)        
        return x_                
            

    def forward(self, x, cond_info):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        skip = []
        skip_sigma = [] #new
        for layer in self.residual_layers:
            x, skip_connection, skip_connection_sigma = layer(x, cond_info) #new: added skip_connection_sigma
            skip.append(skip_connection)
            skip_sigma.append(skip_connection_sigma) #new added this / here
            
        #new
        if "A" in self.mod:
            x_sigma = self.sigma_network_A_conv(skip_sigma)
            x_sigma = x_sigma.reshape(B, K, L)
        elif "B" == self.mod:
            x_sigma = self.sigma_network_B(skip_sigma[0])
        else:
            x_sigma = torch.zeros_like(x)    
        #end new
        
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        return x, x_sigma #new (returning y is new)


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, nheads):
        super().__init__()
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        
        #new
        self.conv2d_layer_sigma = Conv2d_(in_channels=64, out_channels=1, kernel_size=1)
        self.mid_projection_sigma = Conv1d_with_init(channels, channels, 1)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        y = x
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        
        #new
        skip_sigma = y.clone() 
        skip_sigma = self.mid_projection_sigma(skip_sigma)
        skip_sigma = skip_sigma.reshape(base_shape) 
        # skip_sigma = self.conv2d_layer_sigma(skip_sigma)
        # breakpoint()
        skip_sigma = torch.max(skip_sigma, dim=1)[0] #new danger here 
        skip_sigma = skip_sigma.squeeze(1)
        #end new
        
        y = self.mid_projection(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip, skip_sigma #new added skip_sigma


class CSDI_base(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.target_dim = args.keypoint_dim * args.n_major_joints

        self.emb_time_dim = args.model_timeemb
        self.emb_feature_dim = args.model_featureemb
        self.is_unconditional = args.model_is_unconditional

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(args, input_dim, self.emb_total_dim)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else: #new comment: this is the case that happenes in our default case
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def forward(self, batch):
        
        (
            observed_data,
            observed_tp,
            cond_mask
        ) = self.preprocess_data(batch)

        side_info = self.get_side_info(observed_tp, cond_mask)

        B, K, L = observed_data.shape
        
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.close()
        # for t_ in range(B):
        #     test_data = observed_data[t_,:,:50].clone()
        #     test_data = test_data.permute(1,0)
        #     test_data = test_data.cpu().detach().numpy()
        #     test_data = test_data.reshape(50, -1 , 3)
        #     test_data = (test_data+2.5)/(5)
        #     temp = test_data[-1,:,:]
        #     test_data_rl = test_data - temp
        #     temp = test_data[0,:,:]
        #     test_data_rf = test_data - temp
            
        #     #creating 3 subplots with 3 images of test_data and test_data_rf and saving the image:
        #     fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        #     axs[0].imshow(test_data)
        #     axs[1].imshow(test_data_rf)
        #     axs[2].imshow(test_data_rl)
               
        #     plt.savefig('image'+str(t_)+'.png')
        # breakpoint()
        
        
        noisy_data = torch.zeros_like(observed_data).to(self.device)

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info)  # (B,K,L)
        return self.postprocess_data(batch, predicted)


class MY_ST_Transformer(CSDI_base): #new my_
    def __init__(self, args):
        super(MY_ST_Transformer, self).__init__(args) #new my_
        self.Lo = args.obs_frames_num
        self.Lp = args.pred_frames_num

        if args.pre_post_process == 'human3.6m':
            self.preprocess = Human36m_Preprocess(args).to(args.device)
            self.postprocess = Human36m_Postprocess(args).to(args.device)
        elif args.pre_post_process == 'AMASS' or args.pre_post_process == '3DPW':
            self.preprocess = AMASS_3DPW_Preprocess(args).to(args.device)
            self.postprocess = AMASS_3DPW_Postprocess(args).to(args.device)
        else:
            self.preprocess = Preprocess(args).to(args.device)
            self.postprocess = Postprocess(args).to(args.device)

        for p in self.preprocess.parameters():
            p.requires_grad = False

        for p in self.postprocess.parameters():
            p.requires_grad = False
            
        #new:
        self.args = args

    def preprocess_data(self, batch):
        observed_data = batch["observed_pose"].to(self.device)
        observed_data = self.preprocess(observed_data)

        B, L, K = observed_data.shape
        Lp = self.args.pred_frames_num

        observed_data = observed_data.permute(0, 2, 1)  # B, K, L

        observed_data = torch.cat([
            observed_data, torch.zeros([B, K, Lp]).to(self.device)
        ], dim=-1)

        observed_tp = torch.arange(self.Lo + self.Lp).unsqueeze(0).expand(B, -1).to(self.device)
        cond_mask = torch.zeros_like(observed_data).to(self.device)
        cond_mask[:, :, :L] = 1

        return (
            observed_data,
            observed_tp,
            cond_mask
        )

    def postprocess_data(self, batch, predictions): #new used to be predicted

        predicted = predictions[0] #new: this is the poses      
        predicted = predicted[:, :, self.Lo:]
        predicted = predicted.permute(0, 2, 1)
        
        torch.clamp(predicted, min=-1000, max=1000) #new
        
        #new
        sigmas = predictions[1] #this is the sigmas
        
        if self.args['loss']['time_prior'][11:] != "B":
            sigmas = sigmas[:, :, self.Lo:]
        sigmas = sigmas.permute(0, 2, 1)
        

        return {
            'pred_pose': self.postprocess(batch['observed_pose'], predicted),  # B, T, JC
        #new:
            "sigmas": self.postprocess(-1*torch.ones(batch['observed_pose'].shape).to(self.args.device), sigmas, normal=False) #new danger here : the first time I wrote this I added ones instead of zeros
        }