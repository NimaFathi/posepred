import torch
import torch.nn as nn
import numpy as np
import time


def main(opt):
    # training
    for epo in range(start_epoch, opt.epoch + 1):
        ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, opt=opt)

        ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt)

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, opt=None):
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    itera = 1

    for i, (data) in enumerate(data_loader):

        bs, seq_n, feature_n = data.shape

        sup_seq = data.clone()[:, -(seq_in + out_n):, :].reshape([bs, seq_in + out_n, len(feature_n) // 3, 3])
        src = data.clone()
        out_all = net_pred(src, input_n=in_n, output_n=out_n, itera=itera)

        out = out_all[:, seq_in:, 0]
        out = out.reshape([bs, out_n, len(feature_n) // 3, 3])
        data = data.reshape([bs, in_n + out_n, len(feature_n) // 3, 3])
        out_all = out_all.reshape([bs, seq_in + out_n, itera, len(feature_n) // 3, 3])

        # 2d joint loss:
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(out_all[:, :, 0] - sup_seq, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            mpjpe_p3d_h36 = torch.mean(torch.norm(data[:, in_n:] - out, dim=3))

