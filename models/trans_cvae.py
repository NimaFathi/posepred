import torch
import torch.nn as nn
import numpy as np
import pdb

class TRANS_CVAE(nn.Module):

    def __init__(self, args, num_classes=1,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu"):

        super(TRANS_CVAE, self).__init__()
        self.args = args

        # Common parameters
        self.njoints = self.args.keypoints_num
        self.nfeats = self.args.keypoint_dim
        self.num_classes = num_classes

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats


        # Encoder specific
        self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)



        # Decoder specific
        self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)

        self.nframes_out = 16 # Saeed: for now, a fixed value
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)


    def encoder(self, x):

        bs, nframes, nfeats = x.shape

        x = x.permute((1, 0, 2))

        # Saeed: for now, let's put them as fixed
        y = torch.zeros((bs,), dtype=int, device=x.device)#, requires_grad=False)
        mask = torch.ones((bs, nframes), dtype=bool, device=x.device)#, requires_grad=False)

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # adding the mu and sigma queries
        xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1]

        return {"mu": mu, "logvar": logvar}

    def decoder(self, z):

        bs, latent_dim = z.shape
        nframes = self.nframes_out

        # Saeed: for now, let's put them as fixed
        y = torch.zeros((bs,), dtype=int, device=z.device)
        mask = torch.ones((bs, nframes), dtype=bool, device=z.device)

        # shift the latent noise vector to be the action noise
        z = z + self.actionBiases[y]
        z = z[None]  # sequence of size 1

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)

        timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.finallayer(output).reshape(nframes, bs, self.input_feats)

        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 0, 2)

        return output

    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.args.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, inputs):
        pose = inputs['observed_pose']
        bs, frames_n, features_n = pose.shape

        mask = inputs['observed_noise']

        # make data noisy
        if 'observed_noise' in inputs.keys():
            noise = inputs['observed_noise']
        else:
            raise Exception("This model requires noise. set is_noisy to True")
        pose = pose.reshape(bs, frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        noise = noise.reshape(bs, frames_n, self.args.keypoints_num, 1).repeat(1, 1, 1, self.args.keypoint_dim)
        const = (torch.ones_like(noise, dtype=torch.float) * (-1000)).to(self.args.device)
        noisy_pose = torch.where(noise == 1, const, pose).reshape(bs, frames_n, -1)

        # encode
        enc_out = self.encoder(noisy_pose)
        z = self.reparameterize(enc_out)
        
        # decode
        dec_out = self.decoder(z)

        outputs = {'pred_pose': inputs['observed_pose'][:, -1:, :].repeat(1, self.args.pred_frames_num, 1), 'comp_pose': dec_out, 'mask': mask, 'mu': enc_out['mu'], 'logvar': enc_out['logvar']}

        if self.args.use_mask:
            outputs['pred_mask'] = inputs['observed_mask'][:, -1:, :].repeat(1, self.args.pred_frames_num, 1)
        
        return outputs

    def return_latent(self, inputs, seed=None):
        pose = inputs['observed_pose']
        bs, frames_n, features_n = pose.shape

        mask = inputs['noisy_mask']

        # make data noisy
        pose = pose.reshape(bs, frames_n, self.args.keypoints_num, self.args.keypoint_dim)
        mask = mask.reshape(bs, frames_n, self.args.keypoints_num, 1).repeat(1, 1, 1, self.args.keypoint_dim)
        const = (torch.zeros_like(mask, dtype=torch.float) * (-1000)).to(self.args.device)
        noisy_pose = torch.where(mask == 1, const, pose).reshape(bs, frames_n, -1)

        distrib_param = self.encoder(noisy_pose)
        return self.reparameterize(distrib_param, seed=seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, njoints, nfeats, num_frames, num_classes,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 activation="gelu", **kargs):
        super().__init__()
        
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        
        self.input_feats = self.njoints*self.nfeats
        
        self.muQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))
        
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, batch):
        x = batch["observed_pose"]
        bs, nframes, nfeats = x.shape

        x = x.permute((1, 0, 2))

        # Saeed: for now, let's put them as fixed
        y = torch.zeros((bs,))
        mask = torch.ones((bs,nframes))

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # adding the mu and sigma queries
        xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1]

        return {"mu": mu, "logvar": logvar}


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, njoints, nfeats, num_classes,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu", **kargs):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.num_classes = num_classes

        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
                
        self.input_feats = self.njoints*self.nfeats

        self.actionBiases = nn.Parameter(torch.randn(self.num_classes, self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, batch):
        z = batch["z"]

        bs, latent_dim = z.shape[1]
        nframes = self.input_feats

        # Saeed: for now, let's put them as fixed
        y = torch.zeros((bs,))
        mask = torch.ones((bs, nframes))

        # shift the latent noise vector to be the action noise
        z = z + self.actionBiases[y]
        z = z[None]  # sequence of size 1

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output).reshape(nframes, bs, self.input_feats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 0)
        
        return output

