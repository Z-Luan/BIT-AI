from torch import nn
import torch 
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ninp=300, ntoken=150, nhid=2048, nhead=5, nlayers=6, dropout =0.2, embedding_weight=None , embedding_random_flag=False):
        super(Transformer_model, self).__init__()
        self.embedding_random_flag = embedding_random_flag

        self.embed = nn.Embedding(vocab_size , ninp)
        self.embed.weight.data.copy_(embedding_weight)
        self.embed.weight.requires_grad = True
        # self.embed = torch.nn.Embedding.from_pretrained(embedding_weight , freeze = False)

        self.embed_random = nn.Embedding(vocab_size , ninp)
        self.pos_encoder = PositionalEncoding(d_model=ninp, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)

        self.lay1 = nn.Sequential(nn.Linear(ntoken * ninp, 256),
                                  nn.ReLU())
        self.lay2 = nn.Sequential(nn.Linear(256 , 128),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))

        self.lay3 = nn.Linear(128 , 15)

    def forward(self, x):
        if self.embedding_random_flag:
            x = self.embed_random(x)
        else:
            x = self.embed(x)

        # .permute 交换 tensor 维度
        x = x.permute(1, 0, 2)
        # 使输入词向量具有相对位置信息         
        x = self.pos_encoder(x)
        # output = transformer_encoder(inputs)
        # output (max_len, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        
        # 1.展开
        # x = nn.functional.max_pool1d(x.view(1, x.size(2), x.size(1)), kernel_size = x.size(1)).view()
        x = x.contiguous().view(x.size()[0], -1)
        # 2.合成
        # x = torch.mean(x , dim = 1)
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ninp = 300, ntoken = 150,  nhid = 80, nlayers = 1, dropout = 0.2, embedding_weight = None , embedding_random_flag = False):
        super(BiLSTM_model, self).__init__()
        self.embedding_random_flag = embedding_random_flag

        self.embed = nn.Embedding(vocab_size , ninp)
        self.embed.weight.data.copy_(embedding_weight)
        self.embed.weight.requires_grad = True
        # self.embed = torch.nn.Embedding.from_pretrained(embedding_weight , freeze = False)

        self.embed_random = nn.Embedding(vocab_size , ninp)
        self.lstm = nn.LSTM(input_size = ninp, hidden_size = nhid, num_layers = nlayers, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(dropout)


        self.lay1 = nn.Sequential(nn.Linear(2 * ntoken * nhid , 512),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))
        self.lay2 = nn.Sequential(nn.Linear(512 , 256),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))

        self.lay3 = nn.Linear(256 , 15)

    def forward(self, x):
        if self.embedding_random_flag:
            x = self.embed_random(x)
        else:
            x = self.embed(x)

        # output, (hn, cn) = lstm(inputs)
        # output (batch, seq_len, num_directions * hidden_size)
        # h_n (num_layers * num_directions, batch, hidden_size) 储存隐藏状态信息, 即输出信息
        # c_n (num_layers * num_directions, batch, hidden_size) 储存单元状态信息
        x = self.lstm(x)[0]
        x = self.dropout(x)
        # 1.展开
        x = x.contiguous().view(x.size()[0], -1)
        # 2.合成
        # x = torch.mean(x , dim=1)
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x