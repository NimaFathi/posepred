for i, data in enumerate(data_loader):
    bs, in_out_sum, feature_n = data.shape

    sup_seq = data.clone()[:, -(kernel_size + out_n):, :].reshape([bs, kernel_size + out_n, len(feature_n) // 3, 3])
    src = data.clone()

    out_all = net_pred(src, input_n=in_n, output_n=out_n)
    # out.shape: [bs, seq_in + out_n, itera, feature_n]

    out = out_all[:, kernel_size:, 0]
    out = out.reshape([bs, out_n, len(feature_n) // 3, 3])
    data = data.reshape([bs, in_n + out_n, len(feature_n) // 3, 3])
    out_all = out_all.reshape([bs, kernel_size + out_n, itera, len(feature_n) // 3, 3])

    loss_p3d = torch.mean(torch.norm(out_all[:, :, 0] - sup_seq, dim=3))

    mpjpe_p3d_h36 = torch.mean(torch.norm(data[:, in_n:] - out, dim=3))
