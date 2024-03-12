


import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import Linear
from torch_geometric.nn import GCNConv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import numpy as np

author_paperList = 'data/author.pth'
author_later = 'data/author_later.json'
paper_later = 'data/paper_later.json'





batch = 512
epoch = 10
learning_rate = 0.05
lstm_layers_num = 2
lstm_dropout = 0.4
gnn_dropout = 0.
classification_threshold = 0.5


all_data = torch.load(author_paperList)

for i in range(len(all_data)):
    data = all_data[i]
    if data.shape[0] < paperList_len:
        temp = torch.zeros(paperList_len - data.shape[0], 1024)
        all_data[i] = torch.cat([temp, data], dim = 0)
    if data.shape[0] > paperList_len:
        temp = torch.split(data, [data.shape[0] - paperList_len, paperList_len], 
                           dim=0)[-1]
        all_data[i] = temp



data = HeteroData()
data['author'].x = torch.tensor([i for i in range(16249)])
data['paper'].x = torch.tensor([i for i in range(6952)])
data['author'].num_nodes = 16249
data['paper'].num_nodes = 6952

paperList_len = 3

neighbor = 5
neighbor_step = 2
nodeType = 'author'

author2paper = [[], []]
with open(author_later, 'r', encoding='utf-8') as f:
    author_data = json.load(f)
    for i in range(16249):
        for paper in author_data[i]:
            author2paper[0].append(i)
            author2paper[1].append(paper['DI'])


paper2paper = [[], []]
with open(paper_later, 'r', encoding='utf-8') as f:
    paper_data = json.load(f)
    for paper in paper_data:
        for item in paper['CR']:
            paper2paper[0].append(item)
            paper2paper[1].append(paper['DI'])


paper2journal = [[], []]
with open(paper_later, 'r', encoding='utf-8') as f:
    paper_data = json.load(f)
    for paper in paper_data:
        paper2journal[0].append(paper['DI'])
        paper2journal[1].append(paper['SO'])
            
data['author', 'writes', 'paper'].edge_index = torch.tensor(author2paper)
data['paper', 'cite', 'paper'].edge_index = torch.tensor(paper2paper)
data['paper', 'in', 'journal'].edge_index = torch.tensor(paper2journal)

data_undirected=T.ToUndirected()(data)

metapaths = [[('author', 'paper'), ('paper', 'author')], 
             [('author', 'paper'), ('paper', 'paper'),('paper', 'author')],
             [('author', 'paper'), ('paper', 'journal'),('journal', 'paper'),('paper', 'author')],
            ]
data_metapath=T.AddMetaPaths(metapaths=metapaths, 
                             drop_orig_edges=True, 
                             drop_unconnected_nodes=True, 
                             max_sample=20)(data_undirected)

torch.save(data_metapath, 'data/lxx.pth')
data_gnn = torch.load('data/lxx.pth').cuda()


class SemanticAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SemanticAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()


    def forward(self, input, P):
        h = torch.mm(input, self.W)
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0], 1))
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P, -1)
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1, keepdim=True)
        semantic_attentions = F.softmax(semantic_attentions, dim=0)



        att_score_weight =torch.transpose(semantic_attentions,0,1)

        semantic_attentions = semantic_attentions.view(P, 1, 1)


        semantic_attentions = semantic_attentions.repeat(1, N, self.in_features)

        input_embedding = input.view(P, N, self.in_features)

        h_embedding = torch.mul(input_embedding, semantic_attentions)

        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding,att_score_weight


class LHGI(nn.Module):
    def __init__(self, input_dim, att_dim, node_num, dropout_rate=0.):
        super(LHGI, self).__init__()

        self.dropout_rate = dropout_rate
        self.linear = Linear(-1, input_dim)

        self.semantic_level_attention = SemanticAttentionLayer(input_dim, att_dim)
        self.sig = torch.sigmoid

        self.conv_1 = GCNConv(input_dim, input_dim)
        self.conv_2 = GCNConv(input_dim, input_dim)
        self.embedding = torch.nn.Embedding(node_num+1, input_dim)

    def forward(self, graph, nodeType):
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict


        X_origin = x_dict[nodeType]
        X_origin = self.embedding(X_origin)
        X_origin = F.relu(X_origin)

        GNN_conv_list_1 = []

        for edge in edge_index_dict.values():
            X = self.conv_1(X_origin, edge)
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout_rate, training=self.training)

            X = self.conv_2(X, edge)
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout_rate, training=self.training)

            GNN_conv_list_1.append(X)

        muilt_gcn_out = torch.cat(GNN_conv_list_1, dim=0)

        Att_out,att_score_weight = self.semantic_level_attention(muilt_gcn_out, len(edge_index_dict))
        Att_out = Att_out

        return Att_out,att_score_weight


class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()
        self.dropout = lstm_dropout
        self.num_layers = lstm_layers_num
        self.lstm = nn.LSTM(input_size=1024, hidden_size=256, 
                            num_layers=self.num_layers, dropout=self.dropout, 
                            batch_first=True)
        self.linear = nn.Linear(256, 64)
        
        self.attention = nn.MultiheadAttention(embed_dim=64, 
                                            num_heads=4, batch_first=True)
        self.attLinear = nn.Linear(64, 32)
        
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.linear(out)
        attn_output, attn_output_weights = self.attention(out, out, out)
        attn_output = torch.sum(attn_output, dim=1)
        attn_output = self.attLinear(attn_output)
        return attn_output



class endModel(nn.Module):
    def __init__(self):
        super(endModel, self).__init__()
        self.paperList = all_data
        self.linear_gnn = nn.Linear(64, 32)
        self.gnn_dropout = gnn_dropout
        self.gnnModel = LHGI(input_dim=64, att_dim=32, 
                             node_num=data_gnn.num_nodes, 
                             dropout_rate=self.gnn_dropout)
        self.lstmModel = LstmModel()

        self.linear_GnnLstm = nn.Linear(32, 16)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.query = torch.randn(32, 1).cuda()
        self.softmax = nn.Softmax(dim=1)

    def get_gnnPath(self, idx: int):
        loader = NeighborLoader(data_gnn,
                                num_neighbors={key: [neighbor] * neighbor_step for key in data_gnn.edge_types},
                                batch_size=1,
                                directed=True,
                                input_nodes=(nodeType, [idx]),)
        sampled_data = next(iter(loader))
        return sampled_data
    
    def cat_LstmGnn(self, lstmEmbed, gnnEmbed):
        t1 = lstmEmbed @ self.query
        t2 = gnnEmbed @ self.query
        t = self.softmax(torch.cat([t1, t2], dim=1))
        t1 = torch.split(t, 1, dim=1)[0]
        t2 = torch.split(t, 1, dim=1)[1]
        res = lstmEmbed * t1 + gnnEmbed * t2
        return res
        
        
    def forward(self, x):
        author_gnnEmbed = []
        m = True
        for item in torch.split(x, 1, dim=0):
            author1_gnn,att_score_weight = self.gnnModel(self.get_gnnPath(item[0][0].item()), nodeType)
            author1_gnn=author1_gnn[:1]
            author2_gnn,_ = self.gnnModel(self.get_gnnPath(item[0][1].item()), nodeType)
            author2_gnn = author2_gnn[:1]
            author_temp = torch.cat([author1_gnn, author2_gnn], dim=0)
            if m:
                author_gnnEmbed = author_temp.unsqueeze(dim=0)
                m = False
            else:
                author_gnnEmbed = torch.cat([author_gnnEmbed, author_temp.unsqueeze(dim=0)], dim=0)
        author1_gnnEmbed = torch.split(author_gnnEmbed, 1, dim=1)[0].squeeze()
        author2_gnnEmbed = torch.split(author_gnnEmbed, 1, dim=1)[1].squeeze()
        author1_gnnEmbed = self.linear_gnn(author1_gnnEmbed)
        author2_gnnEmbed = self.linear_gnn(author2_gnnEmbed)
        author1_simcseEmbed = []
        f = True
        author2_simcseEmbed = []
        for item in torch.split(x, 1, dim=0):
            author1_lstm = self.paperList[item[0][0].item()]
            author2_lstm = self.paperList[item[0][1].item()]
            if f:
                f = False
                author1_simcseEmbed = author1_lstm.unsqueeze(dim=0)
                author2_simcseEmbed = author2_lstm.unsqueeze(dim=0)
            else:
                author1_simcseEmbed = torch.cat([author1_simcseEmbed, author1_lstm.unsqueeze(dim=0)], dim=0)
                author2_simcseEmbed = torch.cat([author2_simcseEmbed, author2_lstm.unsqueeze(dim=0)], dim=0)
        author1_simcseEmbed = author1_simcseEmbed.cuda()
        author2_simcseEmbed = author2_simcseEmbed.cuda()
        author1_simcseEmbed = self.lstmModel(author1_simcseEmbed)
        author2_simcseEmbed = self.lstmModel(author2_simcseEmbed)

        author1 = self.cat_LstmGnn(author1_simcseEmbed, author1_gnnEmbed)
        author2 = self.cat_LstmGnn(author2_simcseEmbed, author2_gnnEmbed)
        author1 = self.linear_GnnLstm(author1)
        author2 = self.linear_GnnLstm(author2)

        cos_author = self.cos(author1, author2)
        
        return cos_author,att_score_weight

# test_p = 'data/test_p.txt'
# test_n = 'data/test_n.txt'
# train_p = 'data/train_p.txt'
# train_n = 'data/train_n.txt'

class Data(Dataset):
    def __init__(self, data: str, mark: int):
        self.data = self.get_LPData(data)
        self.mark = mark
        
    def get_LPData(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            res = []
            for line in f.readlines():
                data = line.strip('\n').split(' ')
                res.append(data)
            return res
        
    def __getitem__(self, index: int):
        label = self.mark
        author = self.data[index]
        author[0] = int(author[0])
        author[1] = int(author[1])
        res = torch.tensor(author)
        return res, label
    
    def __len__(self):
        return len(self.data)




train_dataset = Data(train_p, 1) + Data(train_n, 0)
test_dataset = Data(test_p, 1) + Data(test_n, 0)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__())
for data in test_dataloader:
    test_datas, test_targets = data
test_datas = test_datas.cuda()
test_targets = test_targets.type(torch.float)

myModel = endModel().cuda()

loss = nn.BCEWithLogitsLoss().cuda()

optim = torch.optim.Adam(myModel.parameters(), lr=learning_rate)

def get_result(myModel,datas,targets):
    myModel.eval()
    with torch.no_grad():
        res, _ = myModel(datas)
        res = res.where(res <= classification_threshold, torch.ones_like(test_targets, dtype=torch.float).cuda())
        res = res.where(res > classification_threshold, torch.zeros_like(test_targets, dtype=torch.float).cuda())
        res = res.cpu().numpy()
        targets = targets.cpu()

        res, att_score_weight = myModel(test_datas)
        res = res.where(res <= classification_threshold, torch.ones_like(test_targets, dtype=torch.float).cuda())
        res = res.where(res > classification_threshold, torch.zeros_like(test_targets, dtype=torch.float).cuda())
        res = res.cpu().numpy()


    my_list = [round(x, 3) for x in att_score_weight[0].tolist()]
    with open('attention_for_matepath.txt', 'a') as file:
        file.write(' '.join(str(number) for number in my_list) + '\n')

for i in range(epoch):
    train_log.write("epoch:{}\n".format(i))
    num = 1
    total_number=len(train_dataset)

    for data in train_dataloader:
        rate_of_progress = num / (total_number / batch)


        myModel.train()
        datas, targets = data
        targets = targets.type(torch.float)
        datas = datas.cuda()
        targets = targets.cuda()
        outputs,att_score_weight = myModel(datas)
        res_loss = loss(outputs, targets)

        my_list = [round(i, 3) for i in att_score_weight[0].tolist()]
        optim.zero_grad()
        res_loss.backward()
        optim.step()
        if (rate_of_progress*100 >90):
            get_result(myModel,datas,targets)
        num += 1







