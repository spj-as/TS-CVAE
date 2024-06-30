import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch import GATConv
import dgl.function as fn 
class GATLayer(nn.Module):
    """
    Reference
    ---------
    https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html?highlight=gat
    """
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim * 3, out_dim, bias=False)  
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        a_input = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        return {'e': self.attn_fc(a_input)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def preprocess_h(self, h):
        h_pad = F.pad(h, (0, 0, 1, 1), 'constant', 0) 
        h_ = torch.cat([h_pad[:-2], h, h_pad[2:]], dim=1)  
        return h_

    def forward(self, g, h):
        h = self.preprocess_h(h)  
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')
    
class CustomGATConv(GATConv):
    def forward(self, graph, feat, get_attention=False):
        graph = graph.local_var()
        if self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. This is '
                               'harmful for some applications, causing silent '
                               'performance regression. Adding self-loops to the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve the issue. '
                               'Setting `allow_zero_in_degree` to be `True` when constructing this module '
                               'will suppress this check and let the code run.')
        feat_src = feat_dst = self.feat_drop(feat)
        if graph.is_block:
            feat_dst = feat_dst[:graph.number_of_dst_nodes()]
        h_src = self.fc(feat_src).view(-1, self._num_heads, self._out_feats)
        h_dst = self.fc(feat_dst).view(-1, self._num_heads, self._out_feats)
        el = (h_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (h_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': h_src, 'el': el})
        graph.dstdata.update({'er': er})
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        if get_attention:
            return rst, graph.edata['a']
        return rst

class CustomGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(CustomGATLayer, self).__init__()
        self.gat_conv = CustomGATConv(in_dim, out_dim, num_heads=num_heads, feat_drop=0.2, attn_drop=0.2, activation=F.elu)

    def forward(self, g, h, get_attention=False):
        h, attn = self.gat_conv(g, h, get_attention=get_attention)
        h = h.view(h.shape[0], -1)  # If multi-head, concatenate the heads
        return h, attn

class StandardGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super(StandardGAT, self).__init__()
        self.layer1 = CustomGATLayer(in_dim, hidden_dim, num_heads)
        self.layer2 = CustomGATLayer(hidden_dim * num_heads, out_dim, 4)

    def forward(self, g, h):
        h, attn1 = self.layer1(g, h, get_attention=True)
        h, attn2 = self.layer2(g, h, get_attention=True)
        
        return h, attn1, attn2

# class StandardGATLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads=2):
#         super(StandardGATLayer, self).__init__()
#         self.num_heads = num_heads
#         self.gat_conv = GATConv(in_dim, out_dim, num_heads=num_heads, feat_drop=0.2, attn_drop=0.2, activation=F.elu)

#     def forward(self, g, h):
#         h = self.gat_conv(g, h)  # h will have shape [batch_size * num_nodes, num_heads, out_dim // num_heads]
        
#         # If multi-head, concatenate the heads
#         if self.num_heads > 1:
#             h = h.view(h.shape[0], -1)  # Concatenate the heads' outputs
#         elif h.dim() == 3 and h.size(1) == 1:
#             h = h.squeeze(1)
        
#         return h

# class StandardGAT(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, num_heads=2):
#         super(StandardGAT, self).__init__()
#         self.layer1 = StandardGATLayer(in_dim, hidden_dim, num_heads)
#         self.layer2 = StandardGATLayer(hidden_dim*num_heads, out_dim)

#     def forward(self, g, h):
#         print(h.shape)
#         h = self.layer1(g, h)
#         print(h.shape)
#         h = self.layer2(g, h)
#         print(h.shape)
#         return h


# class GATLayer_dropedge(nn.Module):
#     """
#     A GAT layer with edge dropout for Graph Attention Networks.
#     """
#     def __init__(self, in_dim, out_dim, dropout_rate):
#         super(GATLayer_dropedge, self).__init__()
#         self.fc = nn.Linear(in_dim * 3, out_dim, bias=False)
#         self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
#         self.dropout_rate = dropout_rate

#     def edge_attention(self, edges):
#         a_input = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
#         e = self.attn_fc(a_input)
#         e_dropout = F.dropout(e, p=self.dropout_rate, training=self.training)
#         return {'e': e_dropout}

#     def message_func(self, edges):
#         return {'z': edges.src['z'], 'e': edges.data['e']}

#     def reduce_func(self, nodes):
#         alpha = F.softmax(nodes.mailbox['e'], dim=1)
#         h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
#         return {'h': h}

#     def preprocess_h(self, h):
#         h_pad = F.pad(h, (0, 0, 1, 1), 'constant', 0)
#         h_ = torch.cat([h_pad[:-2], h, h_pad[2:]], dim=1)
#         return h_

#     def forward(self, g, h):
#         h = self.preprocess_h(h)
#         z = self.fc(h)
#         g.ndata['z'] = z
#         g.apply_edges(self.edge_attention)
#         g.update_all(self.message_func, self.reduce_func)
#         return g.ndata.pop('h')
    

