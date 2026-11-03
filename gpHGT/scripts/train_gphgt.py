"""gpHGT single-cell training, evaluation, and attention export.

Set GPHGT_DATA_DIR to the cohort input directory before running this script.
The model architecture and input conventions follow the supplied implementation.
"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import softmax
import math
import csv
import random
from timm.layers import DropPath
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.loader import DataLoader
datapath = os.path.join(os.environ.get('GPHGT_DATA_DIR', './data'), '')
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
set_seed(42)
print('Current seed fixed. Deterministic mode:', torch.are_deterministic_algorithms_enabled())
# Input data and molecular priors
pathway_gene_dict = {}
with open(datapath + 'pathway_gsva/gene_all.txt', 'r') as f:
    for line in f:
        items = line.strip().split('\t')
        pathway = str(items[0]).replace('"', '')
        genes = items[1:]
        pathway_gene_dict[pathway] = genes
all_genes_path = set()
for genes in pathway_gene_dict.values():
    all_genes_path.update(genes)
print('所有通路中不重复基因数:', len(all_genes_path))
pathway_size_dict = {pw: len(genes) for pw, genes in pathway_gene_dict.items()}
ppi_df = pd.read_csv(datapath + 'HTG/ppi_filtered.txt', sep='\t', header=None, names=['Gene1', 'Gene2', 'Score'])
all_genes = set(ppi_df['Gene1']).union(ppi_df['Gene2'])
for genes in pathway_gene_dict.values():
    all_genes.update(genes)
gene_list = sorted(list(all_genes))
gene2id = {gene: i for i, gene in enumerate(gene_list)}
pathway_list = sorted(list(pathway_gene_dict.keys()))
pathway2id = {pw: i for i, pw in enumerate(pathway_list)}
print(len(gene_list), len(pathway_list))
emb_dict = {}
with open(datapath + '/gene2vec_dim_200_iter_9.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        vec = list(map(float, parts[1:]))
        gene = parts[0]
        emb_dict[gene] = vec
print()
missing_genes = [gene for gene in gene_list if gene not in emb_dict]
print(f'❗缺失预训练向量的基因数: {len(missing_genes)}')
print(missing_genes)
embedding_dim = 200
gene2vec_matrix = []
for gene in gene_list:
    if gene in emb_dict:
        gene2vec_matrix.append(emb_dict[gene])
    else:
        gene2vec_matrix.append(np.zeros(embedding_dim))
print()
gene2vec_matrix_np = np.array(gene2vec_matrix)
gene2vec_tensor = torch.tensor(gene2vec_matrix_np, dtype=torch.float32)
pat_dict = {}
with open(datapath + 'pathway_gsva/keen/KEGG_embeddings.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        path = row[0].replace('"', '').split('pathway:')[1].strip()
        vec = [float(x) for x in row[1:]]
        pat_dict[path] = vec
with open(datapath + 'pathway_gsva/keen/Reactome_embeddings.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        path = row[0].replace('"', '').split('pathway:')[1].strip()
        vec = [float(x) for x in row[1:]]
        pat_dict[path] = vec
with open(datapath + 'pathway_gsva/keen/Wikipathways_embeddings.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        path = row[0].replace('"', '').split('pathway:')[1].strip()
        vec = [float(x) for x in row[1:]]
        pat_dict[path] = vec
print()
missing_pat = [pathway for pathway in pathway_list if pathway not in pat_dict]
print(f'❗缺失预训练向量的基因数: {len(missing_pat)}')
print(missing_pat)
embedding_dim = 200
pat2vec_matrix = []
for pathway in pathway_list:
    if pathway in pat_dict:
        pat2vec_matrix.append(pat_dict[pathway])
    else:
        pat2vec_matrix.append(np.zeros(embedding_dim))
pat2vec_matrix_np = np.array(pat2vec_matrix)
pat2vec_tensor = torch.tensor(pat2vec_matrix_np, dtype=torch.float32)
gene_gene_edge_index = [[gene2id[row['Gene1']], gene2id[row['Gene2']]] for _, row in ppi_df.iterrows() if row['Gene1'] in gene2id and row['Gene2'] in gene2id]
gene_gene_edge_index = torch.tensor(gene_gene_edge_index, dtype=torch.long).T
gene_pathway_edges = []
for pw, genes in pathway_gene_dict.items():
    for gene in genes:
        if gene in gene2id:
            gene_pathway_edges.append([gene2id[gene], pathway2id[pw]])
gene_pathway_edge_index = torch.tensor(gene_pathway_edges, dtype=torch.long).T
scaler_gene = StandardScaler()
gene_exp_train_raw = pd.read_csv(datapath + 'data/NEW_training_sample.csv.gz', compression='gzip', delimiter=',')
rna_names = pd.read_csv(datapath + 'data/rna_names.csv', skiprows=1, header=None)
colnames = rna_names[0].tolist()
gene_exp_train_raw.columns = colnames
gene_exp_train = gene_exp_train_raw.loc[:, gene_exp_train_raw.columns.isin(all_genes)]
gene_exp_train = gene_exp_train[gene_list]
gene_exp_train = pd.DataFrame(scaler_gene.fit_transform(gene_exp_train), index=gene_exp_train.index, columns=gene_exp_train.columns)
gene_exp_test_raw = pd.read_csv(datapath + 'data/NEW_testing_sample.csv.gz', compression='gzip', delimiter=',')
gene_exp_test_raw.columns = colnames
gene_exp_test = gene_exp_test_raw.loc[:, gene_exp_test_raw.columns.isin(all_genes)]
gene_exp_test = gene_exp_test[gene_list]
gene_exp_test = pd.DataFrame(scaler_gene.transform(gene_exp_test), index=gene_exp_test.index, columns=gene_exp_test.columns)
print(gene_exp_train.shape)
scaler_pathway = StandardScaler()
pathway_score = pd.read_csv(datapath + 'pathway_gsva/pathway_all.txt', sep='\t', index_col=0)
pathway_score_train = pathway_score.iloc[:len(gene_exp_train)]
pathway_score_test = pathway_score.iloc[len(gene_exp_train):]
pathway_score_train = pathway_score_train[pathway_list]
pathway_score_test = pathway_score_test[pathway_list]
pathway_score_train = pd.DataFrame(scaler_pathway.fit_transform(pathway_score_train), index=pathway_score_train.index, columns=pathway_score_train.columns)
pathway_score_test = pd.DataFrame(scaler_pathway.transform(pathway_score_test), index=pathway_score_test.index, columns=pathway_score_test.columns)

def build_cell_graph(cell_gene_vector, cell_pathway_vector, gene_gene_edge_index, gene_pathway_edge_index):
    from torch_geometric.data import HeteroData
    data = HeteroData()
    gene_feats = torch.tensor(cell_gene_vector, dtype=torch.float32).unsqueeze(1)
    pathway_feats = torch.tensor(cell_pathway_vector, dtype=torch.float32).unsqueeze(1)
    data['gene'].x = gene_feats
    data['pathway'].x = pathway_feats
    data['gene'].node_id = torch.arange(gene_feats.shape[0], dtype=torch.long)
    data['pathway'].node_id = torch.arange(pathway_feats.shape[0], dtype=torch.long)
    data['gene', 'ppi', 'gene'].edge_index = gene_gene_edge_index
    data['gene', 'in_pathway', 'pathway'].edge_index = gene_pathway_edge_index
    data['pathway', 'has_gene', 'gene'].edge_index = gene_pathway_edge_index.flip(0)
    return data

def build_graph_list(gene_exp, pathway_score, labels, gene_gene_edge_index, gene_pathway_edge_index):
    graphs = []
    for i in range(len(gene_exp)):
        g = build_cell_graph(gene_exp.iloc[i].values, pathway_score.iloc[i].values, gene_gene_edge_index, gene_pathway_edge_index)
        g['label'] = torch.tensor([labels[i]], dtype=torch.long)
        graphs.append(g)
    return graphs

def hetero_to_homo_edge_index(edge_index_dict, x_dict):
    """
    把 HeteroData 的 edge_index_dict 转成一个统一的 edge_index
    """
    node_offset = {}
    offset = 0
    for ntype, x in x_dict.items():
        node_offset[ntype] = offset
        offset += x.size(0)
    edges = []
    for (src, _, dst), eidx in edge_index_dict.items():
        src_offset = node_offset[src]
        dst_offset = node_offset[dst]
        edges.append(torch.stack([eidx[0] + src_offset, eidx[1] + dst_offset], dim=0))
    edge_index = torch.cat(edges, dim=1)
    return edge_index

# Neural-network components
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RoPE(nn.Module):

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = (shape[:-1], shape[-1])
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0
        theta_ks = 1 / base ** (torch.arange(k_max) / k_max)
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class LinearAttention(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution, dim))

    def forward(self, x):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = (qk[0], qk[1])
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = x.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-06)
        kv = k_rope.transpose(-2, -1) * n ** (-0.5) @ (v * n ** (-0.5))
        x = q_rope @ kv * z
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)
        return x

class MLLABlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads=4, mlp_ratio=4.0, qkv_bias=True, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        H = self.input_resolution
        assert L == H, 'input feature has wrong size'
        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).permute(0, 2, 1)
        x = self.act(self.dwc(x)).permute(0, 2, 1)
        x = self.attn(x)
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MLLANodeClusterTransformer(nn.Module):

    def __init__(self, d_model, input_resolution, num_heads=4, mlp_ratio=4.0, num_layers=2):
        super().__init__()
        self.embeddings_dimension = d_model
        self.id_dim = d_model
        self.input_resolution = input_resolution
        self.layers = nn.ModuleList([MLLABlock(dim=d_model * 2, input_resolution=input_resolution, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(num_layers)])
        self.token_type_emb = nn.Embedding(2, d_model)

    def forward(self, node_embeddings, cluster_centers):
        """
        node_embeddings: (N, d)
        cluster_centers: (M, d)
        """
        N, d = node_embeddings.shape
        M = cluster_centers.shape[0]
        node_type_ids = torch.zeros(N, dtype=torch.long, device=node_embeddings.device)
        node_token_emb = self.token_type_emb(node_type_ids)
        h_node = torch.cat([node_embeddings, node_token_emb], dim=-1)
        cluster_type_ids = torch.ones(M, dtype=torch.long, device=cluster_centers.device)
        cluster_token_emb = self.token_type_emb(cluster_type_ids)
        h_cluster = torch.cat([cluster_centers, cluster_token_emb], dim=-1)
        graph_seq = torch.cat([h_node, h_cluster], dim=0).unsqueeze(0)
        x = graph_seq
        for blk in self.layers:
            x = blk(x)
        updated_nodes = x[:, :N, :].squeeze(0)
        return updated_nodes

class GraphAttentionPooling(nn.Module):

    def __init__(self, d_model):
        super(GraphAttentionPooling, self).__init__()
        self.attn_fc = nn.Linear(d_model, 1)

    def forward(self, node_embeddings):
        """
        node_embeddings: (N, d)
        return: graph embedding (1, d)
        """
        attn = self.attn_fc(node_embeddings)
        attn = torch.softmax(attn, dim=0)
        graph_emb = (attn * node_embeddings).sum(dim=0, keepdim=True)
        return graph_emb

class HGTLayer(nn.Module):

    def __init__(self, in_dim, out_dim, n_heads=4, relations=None, dropout=0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim
        self.relations = relations or []
        self.q_linears = nn.ModuleDict()
        self.k_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.rel_pri = nn.ParameterDict()
        self.rel_gate = nn.ParameterDict()
        self.rel_gate_msg = nn.ParameterDict()
        self.out_linear = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.last_attn = {}
        self.res_proj = nn.ModuleDict()

    def add_node_type(self, node_type, in_dim):
        if node_type not in self.q_linears:
            self.q_linears[node_type] = nn.Linear(in_dim, self.out_dim).to(next(self.parameters()).device)
        if node_type not in self.res_proj:
            if in_dim != self.out_dim:
                self.res_proj[node_type] = nn.Linear(in_dim, self.out_dim).to(next(self.parameters()).device)
            else:
                self.res_proj[node_type] = nn.Identity()

    def add_relation(self, rel_key, src_in_dim, dst_in_dim):
        """
        Create K/V for this relation and the static gate and a small gate_mlp based on dst dim.
        """
        if rel_key not in self.k_linears:
            device = next(self.parameters()).device
            self.k_linears[rel_key] = nn.Linear(src_in_dim, self.out_dim).to(device)
            self.v_linears[rel_key] = nn.Linear(src_in_dim, self.out_dim).to(device)
            self.rel_pri[rel_key] = nn.Parameter(torch.ones(self.n_heads, device=device) * 0.1)
            self.rel_gate[rel_key] = nn.Parameter(torch.tensor(1.0, device=device))
            self.rel_gate_msg[rel_key] = nn.Parameter(torch.tensor(1.0, device=device))

    def forward(self, x_dict, edge_index_dict, data=None):
        """
        x_dict: { node_type: [N_t, dim] }
        edge_index_dict: { (src_type, rel, dst_type): [2, E] }
        returns: new_x_dict with same node types but features [N_t, out_dim]
        """
        device = next(self.parameters()).device
        agg_msgs = {ntype: torch.zeros(x.size(0), self.out_dim, device=device) for ntype, x in x_dict.items()}
        attn_dict = {}
        for ntype, x in x_dict.items():
            if ntype not in self.q_linears:
                self.add_node_type(ntype, x.shape[1])
        for rel_triplet, e_idx in edge_index_dict.items():
            if not (isinstance(rel_triplet, tuple) and len(rel_triplet) == 3):
                continue
            src_type, rel_name, dst_type = rel_triplet
            rel_key = '{}__{}__{}'.format(src_type, rel_name, dst_type)
            if rel_key not in self.k_linears:
                self.add_relation(rel_key, x_dict[src_type].shape[1], x_dict[dst_type].shape[1])
        new_x = {ntype: x.new_zeros(x.size(0), self.out_dim) for ntype, x in x_dict.items()}
        for rel_triplet, edge_index in edge_index_dict.items():
            if not (isinstance(rel_triplet, tuple) and len(rel_triplet) == 3):
                continue
            src_type, rel_name, dst_type = rel_triplet
            rel_key = '{}__{}__{}'.format(src_type, rel_name, dst_type)
            src_x = x_dict[src_type].to(device)
            dst_x = x_dict[dst_type].to(device)
            Q_all = self.q_linears[dst_type](dst_x)
            K_all = self.k_linears[rel_key](src_x)
            V_all = self.v_linears[rel_key](src_x)
            Q = Q_all.view(Q_all.size(0), self.n_heads, self.head_dim)
            K = K_all.view(K_all.size(0), self.n_heads, self.head_dim)
            V = V_all.view(V_all.size(0), self.n_heads, self.head_dim)
            src_idx = edge_index[0].to(device)
            dst_idx = edge_index[1].to(device)
            K_e = K[src_idx]
            V_e = V[src_idx]
            Q_e = Q[dst_idx]
            attn_scores = (Q_e * K_e).sum(dim=-1) / math.sqrt(self.head_dim)
            attn_scores = attn_scores * self.rel_pri[rel_key].unsqueeze(0)
            attn_scores = attn_scores * self.rel_gate[rel_key]
            attn_weights = attn_scores.new_zeros(attn_scores.size())
            for h in range(self.n_heads):
                attn_weights[:, h] = softmax(attn_scores[:, h], dst_idx)
            attn_dict[rel_triplet] = attn_weights.detach().cpu()
            V_e_weighted = V_e * attn_weights.unsqueeze(-1)
            V_e_weighted = V_e_weighted * self.rel_gate_msg[rel_key]
            V_e_flat = V_e_weighted.view(V_e_weighted.size(0), -1)
            dst_agg = src_x.new_zeros(dst_x.size(0), self.out_dim)
            dst_agg = dst_agg.index_add(0, dst_idx, V_e_flat)
            new_x[dst_type] = new_x[dst_type] + dst_agg
        out_dict = {}
        for ntype, agg in new_x.items():
            out = self.out_linear(agg)
            out = self.dropout(out)
            src_feat = x_dict[ntype].to(device)
            res = self.res_proj[ntype](src_feat)
            out = self.layer_norm(out + res)
            out_dict[ntype] = F.elu(out)
        self.last_attn = attn_dict
        return out_dict

def gen_rand_label(ft_dict, num_cluster):
    rand_label_dict = dict.fromkeys(ft_dict.keys())
    for k in ft_dict:
        rand_label = torch.randint(num_cluster, (ft_dict[k].shape[0],))
        rand_label = F.one_hot(rand_label, num_cluster).type(dtype=torch.float32)
        rand_label_dict[k] = rand_label
    return rand_label_dict

def att_lpa(adj_dict, init_pseudo_label, target_type=None, num_cluster=10, max_iter=50):
    """
    Attention-guided Label Propagation (跨类型混合聚类版)

    Args:
        adj_dict: {src_type: {dst_type: adj_sparse_tensor}}
        init_pseudo_label: dict {ntype: one-hot [N_ntype, K]}
        attention_dict: HGT注意力，dict结构 {src_type: torch.Tensor(len(neigh_types))}
        target_type: 如果指定，只返回该类型的标签；否则返回所有节点的全局标签
        num_cluster: 簇数
        max_iter: 最大迭代轮次

    Returns:
        labels_all: torch.LongTensor [N]  (全局簇标签)
        或者 {target_type: torch.LongTensor[N_target]} （如果指定了 target_type）
    """
    '\n       Attention-guided LPA\n       现在 edge 的 attention 已经在 adj_dict 构造时融合了，这里直接使用 adj.values()。\n       '
    device = list(init_pseudo_label.values())[0].device
    offsets = {}
    offset = 0
    for ntype, y in init_pseudo_label.items():
        offsets[ntype] = offset
        offset += y.shape[0]
    N = offset
    rows, cols, vals = ([], [], [])
    for src in adj_dict:
        for dst, adj in adj_dict[src].items():
            r, c = adj.indices()
            v = adj.values()
            r_global = r + offsets[src]
            c_global = c + offsets[dst]
            rows.append(r_global)
            cols.append(c_global)
            vals.append(v)
            rows.append(c_global)
            cols.append(r_global)
            vals.append(v)
    row = torch.cat(rows)
    col = torch.cat(cols)
    val = torch.cat(vals)
    A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (N, N), device=device).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
    inv_deg = 1.0 / deg
    r, c = A.indices()
    v = A.values() * inv_deg[r]
    A_hat = torch.sparse_coo_tensor(torch.stack([r, c]), v, (N, N), device=device).coalesce()
    Y_parts = []
    for ntype, Y_nt in init_pseudo_label.items():
        Y_parts.append(Y_nt)
    Y = torch.cat(Y_parts, dim=0).to(device)
    for _ in range(max_iter):
        Y_new = torch.sparse.mm(A_hat, Y)
        Y_new = F.one_hot(Y_new.argmax(dim=1), num_classes=num_cluster).float()
        if torch.equal(Y_new, Y):
            break
        Y = Y_new
    labels_all = Y.argmax(dim=1)
    if target_type is None:
        return labels_all
    else:
        start = offsets[target_type]
        end = start + init_pseudo_label[target_type].shape[0]
        return {target_type: labels_all[start:end]}

# Cell-level heterogeneous graph model
class HGTNet(nn.Module):

    def __init__(self, node_input_dims, hidden_dim=64, n_heads=4, n_layers=2, out_dim=2, gene2vec_tensor=None, pat2vec_tensor=None, intra_gene_layers=1):
        """
        intra_gene_layers: 在完整异质消息传递前，先做多少层仅在 gene->gene (ppi) 上的 intra-gene 更新
        """
        super().__init__()
        self.node_types = list(node_input_dims.keys())
        self.hidden_dim = hidden_dim
        self.intra_gene_layers = intra_gene_layers
        self.num_clusters = 10
        self.T = 10
        self.global_tokens = None
        if gene2vec_tensor is not None:
            self.register_buffer('gene2vec_tensor', gene2vec_tensor)
        else:
            self.register_buffer('gene2vec_tensor', None)
        if pat2vec_tensor is not None:
            self.register_buffer('pat2vec_tensor', pat2vec_tensor)
        else:
            self.register_buffer('pat2vec_tensor', None)
        self.input_proj = nn.ModuleDict()
        for ntype, dim in node_input_dims.items():
            self.input_proj[ntype] = nn.Linear(dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = HGTLayer(in_dim=hidden_dim, out_dim=hidden_dim, n_heads=n_heads, relations=None)
            for nt in self.node_types:
                layer.add_node_type(nt, hidden_dim)
            self.layers.append(layer)
        self.node_cluster_transformer = MLLANodeClusterTransformer(d_model=hidden_dim, input_resolution=len(gene_list) + len(pathway_list) + self.num_clusters, num_heads=4, mlp_ratio=4.0, num_layers=1)
        self.cluster_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pool_attn = nn.ModuleDict({nt: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)) for nt in self.node_types})
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * len(self.node_types), hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(0.4), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, data, save_attention=False):
        device = next(self.parameters()).device
        data = data.to(device)
        x_dict = {}
        for ntype in self.node_types:
            x = data[ntype].x.to(device)
            if ntype == 'pathway' and self.pat2vec_tensor is not None:
                node_ids = data['pathway'].node_id.to(device)
                pat2vec = self.pat2vec_tensor[node_ids].to(device)
                x = torch.cat([x, pat2vec], dim=1)
            if ntype == 'gene' and self.gene2vec_tensor is not None:
                node_ids = data['gene'].node_id.to(device)
                gene2vec = self.gene2vec_tensor[node_ids].to(device)
                x = torch.cat([x, gene2vec], dim=1)
            x = self.input_proj[ntype](x)
            x_dict[ntype] = x
        if self.intra_gene_layers > 0:
            gene_edge_dict = {k: v for k, v in data.edge_index_dict.items() if isinstance(k, tuple) and len(k) == 3 and (k[0] == 'gene') and (k[2] == 'gene')}
            for i in range(min(self.intra_gene_layers, len(self.layers))):
                layer = self.layers[i]
                for rel_triplet in gene_edge_dict.keys():
                    src, rel, dst = rel_triplet
                    rel_key = '{}__{}__{}'.format(src, rel, dst)
                    if rel_key not in layer.k_linears:
                        layer.add_relation(rel_key, x_dict[src].shape[1], x_dict[dst].shape[1])
                x_dict = layer(x_dict, gene_edge_dict, data)
        for j in range(self.intra_gene_layers, len(self.layers)):
            layer = self.layers[j]
            for rel_triplet in data.edge_index_dict.keys():
                if not (isinstance(rel_triplet, tuple) and len(rel_triplet) == 3):
                    continue
                src, rel, dst = rel_triplet
                rel_key = '{}__{}__{}'.format(src, rel, dst)
                if rel_key not in layer.k_linears:
                    layer.add_relation(rel_key, x_dict[src].shape[1], x_dict[dst].shape[1])
            x_dict = layer(x_dict, data.edge_index_dict, data)
        last_layer_attn = None
        for layer in reversed(self.layers):
            if hasattr(layer, 'last_attn') and layer.last_attn:
                last_layer_attn = layer.last_attn
                break
        num_nodes_dict = {ntype: x_dict[ntype].shape[0] for ntype in x_dict}
        adj_dict = {}
        for (src, rel, dst), edge_index in data.edge_index_dict.items():
            src_num, dst_num = (num_nodes_dict[src], num_nodes_dict[dst])
            edge_weights = torch.ones(edge_index.size(1), device=device)
            if last_layer_attn and (src, rel, dst) in last_layer_attn:
                attn = last_layer_attn[src, rel, dst]
                edge_weights = attn.mean(dim=1).detach()
            adj = torch.sparse_coo_tensor(edge_index.to(device), edge_weights.to(device), (src_num, dst_num), device=device).coalesce()
            if src not in adj_dict:
                adj_dict[src] = {}
            adj_dict[src][dst] = adj

        def att_lpa_batch(adj_dict, init_pseudo_label, node_batch_vec, num_clusters=10, max_iter=10):
            """
            Batch-aware label propagation over the big (block-diagonal) graph.
            Args:
                adj_dict: same as before {src: {dst: sparse_adj_tensor (Ns x Nd)}}
                init_pseudo_label: dict {ntype: [N_ntype, K]}
                node_batch_vec: tensor [N_total] with graph id for each global node (same order as concatenation)
                num_clusters: per-graph cluster count
            Returns:
                labels_all: LongTensor [N_total] in [0, num_clusters-1]  (labels per node within its graph)
            """
            device = list(init_pseudo_label.values())[0].device
            offsets = {}
            offset = 0
            parts = []
            for ntype, Y in init_pseudo_label.items():
                offsets[ntype] = offset
                offset += Y.shape[0]
                parts.append(Y)
            N = offset
            rows, cols, vals = ([], [], [])
            for src in adj_dict:
                for dst, adj in adj_dict[src].items():
                    r, c = adj.indices()
                    v = adj.values()
                    r_global = r + offsets[src]
                    c_global = c + offsets[dst]
                    rows.append(r_global)
                    cols.append(c_global)
                    vals.append(v)
                    rows.append(c_global)
                    cols.append(r_global)
                    vals.append(v)
            if len(rows) == 0:
                Y = torch.cat([init_pseudo_label[k] for k in init_pseudo_label], dim=0).to(device)
                return Y.argmax(dim=1)
            row = torch.cat(rows)
            col = torch.cat(cols)
            val = torch.cat(vals)
            A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (N, N), device=device).coalesce()
            deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
            inv_deg = 1.0 / deg
            r_idx, c_idx = A.indices()
            v_norm = A.values() * inv_deg[r_idx]
            A_hat = torch.sparse_coo_tensor(torch.stack([r_idx, c_idx]), v_norm, (N, N), device=device).coalesce()
            Y = torch.cat([init_pseudo_label[k] for k in init_pseudo_label], dim=0).to(device)
            for _ in range(max_iter):
                Y_new = torch.sparse.mm(A_hat, Y)
                Y_new = F.one_hot(Y_new.argmax(dim=1), num_classes=num_clusters).float()
                if torch.equal(Y_new, Y):
                    break
                Y = Y_new
            labels_all = Y.argmax(dim=1)
            return labels_all
        ntypes = list(x_dict.keys())
        node_counts = [x_dict[nt].shape[0] for nt in ntypes]
        batch_parts = []
        for nt in ntypes:
            if hasattr(data[nt], 'batch'):
                batch_parts.append(data[nt].batch.to(device))
            else:
                batch_parts.append(torch.zeros(x_dict[nt].size(0), dtype=torch.long, device=device))
        node_batch_vec = torch.cat(batch_parts, dim=0)
        init_pseudo_label = {k: F.one_hot(torch.randint(self.num_clusters, (ft.shape[0],), device=device), num_classes=self.num_clusters).float() for k, ft in x_dict.items()}
        labels_all = att_lpa_batch(adj_dict, init_pseudo_label, node_batch_vec, num_clusters=self.num_clusters, max_iter=10)
        num_graphs = int(node_batch_vec.max().item() + 1) if node_batch_vec.numel() > 0 else 1
        global_cluster_ids = labels_all + node_batch_vec * self.num_clusters
        num_total_clusters = num_graphs * self.num_clusters
        all_node_embeddings = torch.cat([x_dict[nt] for nt in ntypes], dim=0)
        cluster_embs_all = scatter_mean(all_node_embeddings, global_cluster_ids, dim=0, dim_size=num_total_clusters)
        d = cluster_embs_all.size(1)
        cluster_embs = cluster_embs_all.view(num_graphs, self.num_clusters, d)
        device = all_node_embeddings.device
        ntypes = list(x_dict.keys())
        updated_nodes_list = []
        offset = 0
        for nt in ntypes:
            N = x_dict[nt].shape[0]
            if N > 0:
                updated_nodes_list.append(x_dict[nt])
            offset += N
        all_node_embeddings = torch.cat(updated_nodes_list, dim=0)
        num_graphs = int(node_batch_vec.max().item() + 1) if node_batch_vec.numel() > 0 else 1
        updated_all = torch.zeros_like(all_node_embeddings)
        for g in range(num_graphs):
            mask = node_batch_vec == g
            node_idx = mask.nonzero(as_tuple=True)[0]
            node_emb = all_node_embeddings[node_idx]
            cluster_emb = cluster_embs[g]
            updated_nodes = self.node_cluster_transformer(node_emb, cluster_emb)
            updated_nodes = self.cluster_proj(updated_nodes)
            updated_all[node_idx] = updated_nodes
        offset = 0
        for nt in ntypes:
            N = x_dict[nt].shape[0]
            if N > 0:
                x_dict[nt] = updated_all[offset:offset + N]
            offset += N
        self.last_pool_attn = {}
        pooled = []
        for ntype in self.node_types:
            node_feats = x_dict[ntype]
            if hasattr(data[ntype], 'batch'):
                batch_index = data[ntype].batch.to(node_feats.device)
            else:
                batch_index = torch.zeros(node_feats.size(0), dtype=torch.long, device=node_feats.device)
            attn_logits = self.pool_attn[ntype](node_feats).squeeze(-1)
            attn = torch.zeros_like(attn_logits)
            for g in batch_index.unique():
                mask = batch_index == g
                attn[mask] = torch.softmax(attn_logits[mask], dim=0)
            attn_per_graph = {}
            for g in batch_index.unique():
                g = int(g.item())
                mask = batch_index == g
                local_ids = data[ntype].node_id[mask].cpu().tolist()
                attn_vals = attn[mask].detach().cpu().tolist()
                if ntype == 'gene':
                    names = [gene_list[nid] for nid in local_ids]
                elif ntype == 'pathway':
                    names = [pathway_list[nid] for nid in local_ids]
                else:
                    names = [f'{ntype}_{nid}' for nid in local_ids]
                attn_per_graph[g] = [{'node_id': nid, 'name': name, 'attention': w} for nid, name, w in zip(local_ids, names, attn_vals)]
            self.last_pool_attn[ntype] = attn_per_graph
            num_graphs = int(batch_index.max().item() + 1) if node_feats.size(0) > 0 else 0
            out = node_feats.new_zeros(num_graphs, node_feats.size(1))
            weighted = node_feats * attn.unsqueeze(-1)
            out = out.index_add(0, batch_index, weighted)
            pooled.append(out)
        graph_rep = torch.cat(pooled, dim=1)
        out = self.classifier(graph_rep)
        explain_signals = {'pool_attn': self.last_pool_attn, 'graph_rep': graph_rep.detach().cpu()}
        return (out, graph_rep, explain_signals)

# Evaluation
def evaluate(model, loader, loss_fn=None):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _, _ = model(data)
            prob = F.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            y_true.append(data['label'].view(-1).cpu())
            y_pred.append(pred.cpu())
            y_prob.append(prob[:, 1].cpu())
            if loss_fn is not None:
                loss = loss_fn(out, data['label'].view(-1))
                total_loss += loss.item() * data.num_graphs
                total_samples += data.num_graphs
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_prob = torch.cat(y_prob).numpy()
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    return (avg_loss, acc, auc, y_true, y_pred, y_prob)
# Data partitions and graph loaders
y_train_full = pd.read_csv(datapath + 'data/NEW_training_label.csv.gz', compression='gzip', delimiter=',').iloc[:, 0].values
y_test = pd.read_csv(datapath + 'data/NEW_testing_label.csv.gz', compression='gzip', delimiter=',').iloc[:, 0].values
exp_train, exp_val, pw_train, pw_val, y_train, y_val = train_test_split(gene_exp_train, pathway_score_train, y_train_full, test_size=4310, random_state=42, stratify=y_train_full)
cell_graphs_train = build_graph_list(exp_train, pw_train, y_train, gene_gene_edge_index, gene_pathway_edge_index)
cell_graphs_val = build_graph_list(exp_val, pw_val, y_val, gene_gene_edge_index, gene_pathway_edge_index)
cell_graphs_test = build_graph_list(gene_exp_test, pathway_score_test, y_test, gene_gene_edge_index, gene_pathway_edge_index)
train_loader = DataLoader(cell_graphs_train, batch_size=32, shuffle=True)
val_loader = DataLoader(cell_graphs_val, batch_size=32, shuffle=False)
test_loader = DataLoader(cell_graphs_test, batch_size=32, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
node_input_dims = {'gene': 1 + (gene2vec_tensor.size(1) if gene2vec_tensor is not None else 0), 'pathway': 1 + (pat2vec_tensor.size(1) if pat2vec_tensor is not None else 0)}
model = HGTNet(node_input_dims=node_input_dims, hidden_dim=64, n_heads=2, n_layers=3, out_dim=2, gene2vec_tensor=gene2vec_tensor, pat2vec_tensor=pat2vec_tensor, intra_gene_layers=1).to(device)
data = next(iter(train_loader)).to(device)
print(model.input_proj['gene'].weight.shape)
print(model.input_proj['pathway'].weight.shape)
print(data['gene'].x.shape)
print(data['gene'].node_id.shape)
print(data['pathway'].x.shape)
print(data['pathway'].node_id.shape)
print(f'Model device: {next(model.parameters()).device}')
print(f"Data device before: {data['gene'].x.device}")
data = data.to(device)
print(f"Data device after: {data['gene'].x.device}")
for k, v in data.edge_index_dict.items():
    print(f'Edge {k} device: {v.device}')
# Optimization and checkpoint selection
loss_fn = nn.CrossEntropyLoss()
num_epochs = 20
num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = int(0.02 * num_training_steps)
# Register relation projections and gates before constructing the optimizer.
for layer_index, layer in enumerate(model.layers):
    relation_types = data.edge_index_dict.keys()
    if layer_index < model.intra_gene_layers:
        relation_types = [relation for relation in relation_types
                          if relation[0] == "gene" and relation[2] == "gene"]
    for source_type, relation_name, target_type in relation_types:
        relation_key = f"{source_type}__{relation_name}__{target_type}"
        layer.add_relation(relation_key, model.hidden_dim, model.hidden_dim)

os.makedirs(os.path.join(datapath, "results", "models"), exist_ok=True)
checkpoint_path = os.path.join(datapath, "results", "models", "last_model_msg.pt")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.001)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
best_val_loss = float('inf')
early_stop_counter = 0
step = 0
patience = 50
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
        data = data.to(device)
        out, _, _ = model(data)
        loss = loss_fn(out, data['label'].view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step()
        step += 1
    val_loss, val_acc, val_auc, _, _, _ = evaluate(model, val_loader, loss_fn=nn.CrossEntropyLoss())
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f},Val Loss:{val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
    print(f'Learning rate after epoch {epoch + 1}: {scheduler.get_last_lr()[0]:.6f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print('Early stopping triggered.')
            break
# Save and evaluate the final completed training epoch.
torch.save(model.state_dict(), checkpoint_path)
test_loss, test_acc, test_auc, y_true_test, y_pred_test, y_prob_test = evaluate(model, test_loader, loss_fn=nn.CrossEntropyLoss())
final_msg = f'\n Final Test Loss: {test_loss:.4f},Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}'
print(final_msg)
df = pd.DataFrame({'TrueLabel': y_true_test, 'PredLabel': y_pred_test, 'PredProb': y_prob_test})
df.to_csv(datapath + 'results/per_cell_predictions_msg.csv', index=False)
import json
import os

# Attention and representation export
def save_explain_signals_batch(batch_id, labels, signals, node_names, outdir):
    os.makedirs(outdir, exist_ok=True)
    save_obj = {'batch_id': batch_id, 'labels': labels, 'pool_attn': signals['pool_attn'], 'node_names': node_names, 'graph_rep': signals['graph_rep'].numpy().tolist()}
    out_fp = os.path.join(outdir, f'batch{batch_id}.json')
    with open(out_fp, 'w') as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        out, graph_rep, signals = model(data, save_attention=True)
        labels = data['label'].view(-1).cpu().numpy().tolist()
        node_names = {'gene': gene_list, 'pathway': pathway_list}
        save_explain_signals_batch(i, labels, signals, node_names, datapath + 'results/signals_msg')
