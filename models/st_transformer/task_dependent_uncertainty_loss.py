
import numpy as np
import torch
import torch.nn as nn

dim_used = np.array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
       63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
       80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95])

class TDUncertaintyLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.action_aware:
            self.action_list = ["walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing",
                              "purchases", "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog",
                              "walkingtogether"]
            self.action_map = {self.action_list[i]: i for i in range(len(self.action_list))}


    def proc(self, observed_pose):
        # return observed_pose[:, :, dim_used]
        return observed_pose


    def forward(self, y_pred, y_true):
      
        sigma = y_pred['sigma']
        # actions =  y_true['action']
        y_pred = self.proc(y_pred['pred_pose']) # B,T,JC
        y_true = self.proc(y_true['future_pose']) # B,T,JC
        # y_pred = y_pred['pred_pose']
        # y_true = y_true['future_pose']

        B,T,JC = y_pred.shape
        C = 3
        J = JC//C

        y_pred = y_pred.view(B, T, J, C)
        y_true = y_true.view(B, T, J, C)

        if self.args.action_aware:
            # indx = torch.tensor([self.action_map[act] for act in actions]).to(self.args.device)
            # sigma = sigma(indx) # B,TJ or B,J or B,T

            # TDUL_T = T if self.args.time_aware else 1
            # TDUL_J = J if self.args.joint_aware else 1
            # sigma = sigma.view(-1, TDUL_T, TDUL_J)
            pass

        elif sigma.dim == 2:
            sigma = sigma.unsqueeze(0) # 1,T,J or 1,1,J or 1,T,1

        l = torch.norm(y_pred - y_true, dim=-1) # B,T,J

        if self.args.squared:
            l = l * l

        l = torch.mean(torch.exp(-sigma) * l + sigma)

        return {
          'loss' : l
        }

##############################################################################################################################

# import numpy as np
# import torch
# import torch.nn as nn

# dim_used = np.array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
#        26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
#        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
#        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
#        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95])



# class TDUncertaintyLoss(nn.Module):

#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         if args.action_aware:
#             self.action_list = ["walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing",
#                               "purchases", "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog",
#                               "walkingtogether"]
#             self.action_map = {self.action_list[i]: i for i in range(len(self.action_list))}

#         # self.moving_avg = torch.zeros(1, 25, dim_used.shape[0] // 3, requires_grad=False)
#         # self.init_moving_avg = True


#     def proc(self, observed_pose):
#         # return observed_pose[:, :, dim_used]
#         return observed_pose


#     def forward(self, y_pred, y_true, Test=False):
      
#         # sigma = y_pred['sigma']
#         # actions =  y_true['action']
#         y_pred = self.proc(y_pred['pred_pose']) # B,T,JC
#         y_true = self.proc(y_true['future_pose']) # B,T,JC
#         # y_pred = y_pred['pred_pose']
#         # y_true = y_true['future_pose']

#         B,T,JC = y_pred.shape
#         C = 3
#         J = JC//C

#         y_pred = y_pred.view(B, T, J, C)
#         y_true = y_true.view(B, T, J, C)

#         # if self.args.action_aware:
#         #     indx = torch.tensor([self.action_map[act] for act in actions]).to(self.args.device)
#         #     sigma = sigma(indx) # B,TJ or B,J or B,T

#         #     TDUL_T = T if self.args.time_aware else 1
#         #     TDUL_J = J if self.args.joint_aware else 1
#         #     sigma = sigma.view(-1, TDUL_T, TDUL_J)

#         # elif sigma.dim == 2:
#         #     sigma = sigma.unsqueeze(0) # 1,T,J or 1,1,J or 1,T,1

#         l = torch.norm(y_pred - y_true, dim=-1) # B,T,J
#         # batchNormed = torch.unsqueeze(torch.mean(l.detach(), dim=0), 0)
        
#         # if not Test:
#         #     if self.init_moving_avg:
#         #         self.moving_avg = batchNormed
#         #         self.init_moving_avg = False

#         #     self.moving_avg = self.moving_avg *  0.9 + batchNormed * 0.1
        
#         # l = torch.mean(l / (self.moving_avg + 1e-5))
        
#         l = torch.mean(l / (torch.unsqueeze(torch.mean(l.detach(), dim=0), 0) + 1e-5))

#         return {
#           'loss' : l
#         }

# # ##################################################################################################
# # ##################################################################################################

# # # import numpy as np
# # # import torch
# # # import torch.nn as nn

# # # dim_used = np.array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
# # #        26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
# # #        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
# # #        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
# # #        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95])



# # # class TDUncertaintyLoss(nn.Module):

# # #     def __init__(self, args):
# # #         super().__init__()
# # #         self.args = args

# # #         if args.action_aware:
# # #             self.action_list = ["walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing",
# # #                               "purchases", "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog",
# # #                               "walkingtogether"]
# # #             self.action_map = {self.action_list[i]: i for i in range(len(self.action_list))}


# # #     def proc(self, observed_pose):
# # #         return observed_pose[:, :, dim_used]


# # #     def forward(self, y_pred, y_true, Test=False):
      
# # #         # sigma = y_pred['sigma']
# # #         # actions =  y_true['action']
# # #         y_pred = self.proc(y_pred['pred_pose']) # B,T,JC
# # #         y_true = self.proc(y_true['future_pose']) # B,T,JC
# # #         # y_pred = y_pred['pred_pose']
# # #         # y_true = y_true['future_pose']

# # #         B,T,JC = y_pred.shape
# # #         C = 3
# # #         J = JC//C

# # #         y_pred = y_pred.view(B, T, J, C)
# # #         y_true = y_true.view(B, T, J, C)

# # #         # if self.args.action_aware:
# # #         #     indx = torch.tensor([self.action_map[act] for act in actions]).to(self.args.device)
# # #         #     sigma = sigma(indx) # B,TJ or B,J or B,T

# # #         #     TDUL_T = T if self.args.time_aware else 1
# # #         #     TDUL_J = J if self.args.joint_aware else 1
# # #         #     sigma = sigma.view(-1, TDUL_T, TDUL_J)

# # #         # elif sigma.dim == 2:
# # #         #     sigma = sigma.unsqueeze(0) # 1,T,J or 1,1,J or 1,T,1

# # #         l = torch.norm(y_pred - y_true, dim=-1) # B,T,J
# # #         l = torch.mean(l / (torch.unsqueeze(torch.mean(l.detach(), dim=0), 0) + 1e-5))

# # #         # if self.args.squared:
# # #         #     l = l * l

# # #         # l = torch.mean(torch.exp(-sigma) * l + sigma)

# # #         return {
# # #           'loss' : l
# # #         }

