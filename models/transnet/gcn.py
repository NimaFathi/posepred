import torch
from torch import Graph, nn
import math 

class GraphConvolution(nn.Module):
  """Implements graph convolutions."""

  def __init__(self, in_features, out_features, n_nodes, bias=False):

    super(GraphConvolution, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.n_nodes = n_nodes
    self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

    self.att = nn.Parameter(torch.FloatTensor(n_nodes, n_nodes))
    if bias:
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
    else:
        self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    # stdv = 1. / math.sqrt(self.weight.size(1))
    # self.weight.data.uniform_(-stdv, stdv)
    # self.att.data.uniform_(-stdv, stdv)
    # if self.bias is not None:
    #     self.bias.data.uniform_(-stdv, stdv)
    nn.init.xavier_uniform_(self.weight.data)
    nn.init.xavier_uniform_(self.att.data)
    if self.bias is not None:
      nn.init.xavier_uniform_(self.bias.data)

  def forward(self, x):
    """Forward pass.

    Args:
      x -- [..., n_nodes, input_features]
    Returns:
      output -- Shape is [..., n_nodes, output_features].
    """
    support = torch.matmul(x, self.weight)
    output = torch.matmul(self.att, support)

    if self.bias is not None:
        return output + self.bias
    else:
        return output

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
             + str(self.in_features) + ' -> ' \
             + str(self.out_features) + ')'



class GCN(nn.Module):
  """Implements graph convolutional network."""

  def __init__(self, in_features, out_features, hidden_features, n_nodes, n_hidden_layers, bias=False):
    super(GCN, self).__init__()

    if n_hidden_layers == 0:
      layers = [
        GraphConvolution(in_features, out_features, n_nodes, bias), 
        nn.BatchNorm2d(50),        
        nn.ReLU()]

    else:
      layers = [
        GraphConvolution(in_features, hidden_features, n_nodes, bias),
        #nn.BatchNorm1d(hidden_features),
        nn.ReLU()]

      for _ in range(n_hidden_layers):
        layers.extend([
          GraphConvolution(hidden_features, hidden_features, n_nodes, bias), 
          nn.BatchNorm2d(50),          
          nn.ReLU()])

      layers.extend([
        GraphConvolution(hidden_features, out_features, n_nodes, bias), 
        nn.BatchNorm2d(50),        
        nn.ReLU()])

    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    """Forward pass.

    Args:
      x -- [..., n_nodes, input_features]
    Returns:
      output -- Shape is [..., n_nodes, output_features].
    """
    return self.layers(x)
