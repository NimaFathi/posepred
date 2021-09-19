import torch


def run_model(net_pred):
    in_n = input_n
    out_n = output_n
    seq_in = kernel_size
    itera = 1

    for i, data in enumerate(data_loader):
        bs, seq_n, feature_n = data.shape

        sup_seq = data.clone()[:, -(seq_in + out_n):, :].reshape([bs, seq_in + out_n, len(feature_n) // 3, 3])
        src = data.clone()

        out_all = net_pred(src, input_n=in_n, output_n=out_n, itera=itera)

        out = out_all[:, seq_in:, 0]
        out = out.reshape([bs, out_n, len(feature_n) // 3, 3])
        data = data.reshape([bs, in_n + out_n, len(feature_n) // 3, 3])
        out_all = out_all.reshape([bs, seq_in + out_n, itera, len(feature_n) // 3, 3])

        loss_p3d = torch.mean(torch.norm(out_all[:, :, 0] - sup_seq, dim=3))

        mpjpe_p3d_h36 = torch.mean(torch.norm(data[:, in_n:] - out, dim=3))
