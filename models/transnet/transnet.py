import torch
from torch import nn
import math
from models.transnet.pose_gcn import SimpleEncoder
from models.transnet.gcn import GCN

class TransNet(nn.Module):
    def __init__(self, args):
        super(TransNet, self).__init__()
        self.args = args

        self.embedding1 = nn.Linear(args.keypoints_num * args.keypoint_dim, args.d_model)
        # self.embedding1 = GCN(
        #     in_features=args.keypoint_dim, 
        #     out_features=args.d_model//args.keypoints_num, 
        #     hidden_features=None, 
        #     n_nodes=args.keypoints_num, 
        #     n_hidden_layers=0, 
        #     bias=False)

        self.transformers = nn.ModuleList(
            [nn.Transformer(
            d_model=args.d_model, 
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            batch_first=True) for _ in range(4)]
        )
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm2d(25) for _ in range(3)]
        )


        #self.fc2 = nn.Linear(args.d_model, args.keypoints_num * args.keypoint_dim)

    def forward(self, inputs):
        B, T_src, N = inputs['observed_pose'].shape
        _, T_tgt, _ = inputs['future_pose'].shape
        J = self.args.keypoints_num
        D = self.args.keypoint_dim
        assert J * D == N

        x = inputs['observed_pose'] #.reshape(B, T_src, J, D)
        y = inputs['future_pose'] #.reshape(B, T_tgt, J, D)
        src = torch.relu(self.embedding1(x))
        tgt = torch.relu(self.embedding1(y))

        #src = src.reshape(B, T_src, J*D)
        #tgt = tgt.reshape(B, T_tgt, J*D)
        out = self.transformers[0](src, tgt) + x[:, 49, :].unsqueeze(1).repeat(1, 25, 1)
        for i, tblock in enumerate(self.transformers[1:]):
            self.batchnorms[i]
            torch.relu(out)
            out = tblock(out, tgt) + out
            

        #src = inputs['observed_pose']
        #tgt = inputs['future_pose']
        #out = self.transformer(src, tgt)

        
        outputs = {
                'pred_pose': out
                }
        
        return outputs

# class PositionalEncoding(nn.Module):
#     def __init__(self, dim_model, dropout_p, max_len):
#         super().__init__()
#         # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#         # max_len determines how far the position can have an effect on a token (window)
        
#         # Info
#         self.dropout = nn.Dropout(dropout_p)
        
#         # Encoding - From formula
#         pos_encoding = torch.zeros(max_len, dim_model)
#         positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
#         division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
#         # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
#         pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
#         # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
#         pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
#         # Saving buffer (same as parameter without gradients needed)
#         pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pos_encoding",pos_encoding)
        
#     def forward(self, token_embedding: torch.tensor) -> torch.tensor:
#         # Residual connection + pos encoding
#         return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


# class Transformer(nn.Module):
#     """
#     Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
#     Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
#     """
#     # Constructor
#     def __init__(
#         self,
#         dim_model,
#         num_heads,
#         num_encoder_layers,
#         num_decoder_layers,
#         dropout_p,
#     ):
#         super().__init__()

#         # INFO
#         self.model_type = "Transformer"
#         self.dim_model = dim_model

#         # LAYERS
#         self.positional_encoder = PositionalEncoding(
#             dim_model=dim_model, dropout_p=dropout_p, max_len=5000
#         )
#         self.embedding = nn.Linear(args.keypoints_num * args.keypoint_dim, args.d_model)
#         self.transformer = nn.Transformer(
#             d_model=dim_model,
#             nhead=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dropout=dropout_p,
#         )
#         self.out = nn.Linear(args.d_model, args.keypoints_num * args.keypoint_dim)
        
#     def forward(self, inputs, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
#         # Src size must be (batch_size, src sequence length)
#         # Tgt size must be (batch_size, tgt sequence length)

#         # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
#         src = inputs['observed_pose'] * math.sqrt(self.dim_model)
#         tgt = inputs['future_pose'] * math.sqrt(self.dim_model)
#         src = self.positional_encoder(src)
#         tgt = self.positional_encoder(tgt)
        
#         # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
#         # to obtain size (sequence length, batch_size, dim_model),
#         src = src.permute(1,0,2)
#         tgt = tgt.permute(1,0,2)

#         # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
#         transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
#         out = self.out(transformer_out)
        
#         return out
      
#     def get_tgt_mask(self, size) -> torch.tensor:
#         # Generates a squeare matrix where the each row allows one word more to be seen
#         mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
#         mask = mask.float()
#         mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
#         mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
#         # EX for size=5:
#         # [[0., -inf, -inf, -inf, -inf],
#         #  [0.,   0., -inf, -inf, -inf],
#         #  [0.,   0.,   0., -inf, -inf],
#         #  [0.,   0.,   0.,   0., -inf],
#         #  [0.,   0.,   0.,   0.,   0.]]
        
#         return mask
    
#     def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
#         # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
#         # [False, False, False, True, True, True]
#         return (matrix == pad_token)

