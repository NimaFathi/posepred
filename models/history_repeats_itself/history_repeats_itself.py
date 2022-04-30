import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np
import math

# from models.history_repeats_itself.utils import get_dct_matrix

from models.history_repeats_itself.utils import data_utils, util


class HistoryRepeatsItself(nn.Module):
    def __init__(self, args):
        super(HistoryRepeatsItself, self).__init__()
        self.args = args
        self.device = args.device
        self.net_pred = AttModel(in_features=args.in_features, kernel_size=args.kernel_size, d_model=args.d_model,
                                 num_stage=args.num_stage, dct_n=args.dct_n, device=self.device)
        # if is_train == 0:
        #     net_pred.train()
        # else:
        #     net_pred.eval()
        l_p3d = 0
        # if is_train <= 1:
        #     m_p3d_h36 = 0
        # else:
        #     titles = np.array(range(opt.output_n)) + 1
        #     m_p3d_h36 = np.zeros([opt.output_n])
        # n = 0
        # todo
        self.in_n = args.input_n
        self.out_n = args.output_n
        if 'sig5' in args.un_mode:
            un_params = torch.nn.Parameter(torch.zeros(self.args.in_features//3, 5))
        elif 'sigstar' in args.un_mode:
            un_params = torch.nn.Parameter(torch.zeros(self.args.in_features//3, 2))
        else:
            un_params = torch.nn.Parameter(torch.zeros(15, self.out_n + self.args.kernel_size ,self.args.in_features//3))

        self.un_params = un_params
        # torch.nn.init.xavier_uniform_(self.un_params)
        torch.nn.init.normal_(self.un_params, mean=1.5, std=0.2)
        self.args.loss.un_mode = self.args.un_mode

        self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                  26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                  46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                  75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        self.seq_in = args.kernel_size
        self.sample_rate = 2
        # joints at same loc
        self.joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        self.index_to_ignore = np.concatenate(
            (self.joint_to_ignore * 3, self.joint_to_ignore * 3 + 1, self.joint_to_ignore * 3 + 2))
        self.joint_equal = np.array([13, 19, 22, 13, 27, 30])
        self.index_to_equal = np.concatenate((self.joint_equal * 3, self.joint_equal * 3 + 1, self.joint_equal * 3 + 2))
        self.itera = 1
        self.idx = np.expand_dims(np.arange(self.seq_in + self.out_n), axis=1) + (
                self.out_n - self.seq_in + np.expand_dims(np.arange(self.itera), axis=0))

    # @staticmethod
    # def exp2xyz(inputs):
    #     is_cuda_type = False
    #     if inputs.is_cuda:
    #       is_cuda_type = True
    #     # print(inputs.shape)
    #     b,n, d = inputs.shape
    #     the_sequence = np.array(inputs.cpu())
    #     the_sequence = np.reshape(the_sequence, (-1, the_sequence.shape[-1]))
    #     the_sequence = torch.from_numpy(the_sequence).float().cpu() #todo
    #     # remove global rotation and translation
    #     the_sequence[:, 0:6] = 0
    #     p3d = data_utils.expmap2xyz_torch(the_sequence)
    #     p3d = np.reshape(p3d,(b, n, -1))
    #     if is_cuda_type:
    #       p3d = p3d.cuda()
    #     # print(p3d.shape)
    #     return p3d

    def forward(self, inputs):
        seq = torch.cat((inputs['observed_pose'], inputs['future_pose']), dim=1)
        # p3d_h36 = self.exp2xyz(seq)
        p3d_h36 = seq.reshape(seq.shape[0], seq.shape[1], -1)
        # print('kk', p3d_h36.shape)
        # print(i)
        batch_size, seq_n, _ = p3d_h36.shape
        p3d_h36 = p3d_h36.float()  # todo
        p3d_sup = p3d_h36.clone()[:, :, self.dim_used][:, -self.out_n - self.seq_in:].reshape(
            [-1, self.seq_in + self.out_n, len(self.dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, self.dim_used]
        p3d_out_all = self.net_pred(p3d_src, input_n=self.in_n, output_n=self.out_n, itera=self.itera)
        p3d_out = p3d_h36.clone()[:, self.in_n:self.in_n + self.out_n]
        # print(self.dim_used, self.seq_in, p3d_out_all.shape, p3d_out.shape)
        p3d_out[:, :, self.dim_used] = p3d_out_all[:, self.seq_in:, 0]
        p3d_out[:, :, self.index_to_ignore] = p3d_out[:, :, self.index_to_equal]
        p3d_out = p3d_out.reshape([-1, self.out_n, 96])

        # p3d_h36 = p3d_h36.reshape([-1, self.in_n + self.out_n, 32, 3])

        p3d_out_all = p3d_out_all.reshape(
            [batch_size, self.seq_in + self.out_n, self.itera, len(self.dim_used) // 3, 3])

        return {'pred_pose': p3d_out_all, 'pred_metric_pose': p3d_out, 'un_params': self.un_params}


class AttModel(nn.Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10, device='cpu'):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        self.device = device
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                       num_stage=num_stage,
                       node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        # print('src', src.shape)
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.device)  # todo
        idct_m = torch.from_numpy(idct_m).float().to(self.device)  # todo

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        # print('ss', src_value_tmp.shape, dct_m[:dct_n].unsqueeze(dim=0).shape)
        # print('dd', bs, vn, dct_n)
        # print('m', torch.matmul(dct_m[:dct_n].unsqueeze(dim=0).cpu(), src_value_tmp.cpu()).shape)
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))
            if itera > 1:
                # update key-value query
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)

                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)

                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)

                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        outputs = torch.cat(outputs, dim=2)
        # print('out', outputs.shape)
        return outputs


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y

# '''
# C:\Users\samen\Desktop\term7\b\git_pose\datasets
# python -m api.preprocess dataset=human3.6m official_annotation_path=C:\Users\samen\Desktop\term7\b\git_pose\datasets\Poses\ data_type=validation keypoint_dim=3 skip_num=1 obs_frames_num=50 pred_frames_num=10 interactive=false
#
# '''
# python -m api.train model=history_repeats_itself keypoint_dim=3 train_dataset=C:\Users\samen\Desktop\term7\b\git_pose\preprocessed_data\human36m\train_50_10_1_human3.6m.jsonl valid_dataset=C:\Users\samen\Desktop\term7\b\git_pose\preprocessed_data\human36m\validation_50_10_1_human3.6m.jsonl epochs=10 data.shuffle=True device=cpu snapshot_interval=10 hydra.run.dir=.\outputs\21
#
# '''
# python -m api.train model=hisroty_repeats_itself keypoint_dim=3 train_dataset=C:\Users\samen\Desktop\term7\b\git_pose\preprocessed_data\human36m\train_50_10_1_human3.6m.jsonl valid_dataset=C:\Users\samen\Desktop\term7\b\git_pose\preprocessed_data\human36m\validation_50_10_1_human3.6m.jsonl epochs=10 data.shuffle=True device=cpu snapshot_interval=10 hydra.run.dir=./

# python -m api.train model=history_repeats_itself keypoint_dim=3 train_dataset=C:\Users\samen\Desktop\term7\b\git_pose\preprocessed_data\human36m\train_50_10_1_human3.6m.jsonl valid_dataset=C:\Users\samen\Desktop\term7\b\git_pose\preprocessed_data\human36m\validation_50_10_1_human3.6m.jsonl epochs=1 data.shuffle=True device=cpu snapshot_interval=10 hydra.run.dir=.\outputs\59
#111
# '''
