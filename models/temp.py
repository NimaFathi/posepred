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

        bs, seq_n, _ = data.shape

        sup = data.clone()[:, -(out_n + seq_in):, :].reshape([bs, seq_in + out_n, len(in_n) // 3, 3])
        src = data.clone()
        out_all = net_pred(src, input_n=in_n, output_n=out_n, itera=itera)

        p3d_out = data.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = out_all[:, seq_in:, 0]
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])

        data = data.reshape([-1, in_n + out_n, 32, 3])

        out_all = out_all.reshape([bs, seq_in + out_n, itera, len(dim_used) // 3, 3])

        # 2d joint loss:

        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(out_all[:, :, 0] - sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * bs

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(data[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * bs
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(data[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

    return ret
