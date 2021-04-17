import torch
import torch.nn as nn


class CategoricalEmbeds(nn.Module):
    def __init__(self, cardinality, emb_sz, emb_dropout):
        super(CategoricalEmbeds, self).__init__()
        self.cardinality = cardinality
        self.emb_sz = emb_sz
        self.emb_dropout = emb_dropout
        self.emb_layer = nn.Embedding(self.cardinality, self.emb_sz)
        self.dropouts = nn.Dropout(self.emb_dropout)

    def forward(self, data, col_num):
        x = self.emb_layer(data[:, col_num])
        x = self.dropouts(x)
        return x


class TorchTabular(nn.Module):
    def __init__(self, embed_details, dropouts, linear_layer_sz, output_layer_sz):
        super(TorchTabular, self).__init__()
        self.embed_layers = nn.ModuleList([CategoricalEmbeds(cardinality=c, emb_sz=d, emb_dropout=dropouts)
                                           for c, d in embed_details])
        self.total_embedding_sz = sum([dim for _, dim in embed_details])
        self.bnorm_emb = nn.BatchNorm1d(self.total_embedding_sz)
        self.feed_fwd = nn.Sequential(
            nn.Linear(self.total_embedding_sz, linear_layer_sz),
            nn.ReLU(),
            nn.Dropout(dropouts),
            nn.BatchNorm1d(linear_layer_sz),
            nn.Linear(linear_layer_sz, linear_layer_sz),
            nn.ReLU(),
            nn.Dropout(dropouts),
            nn.BatchNorm1d(linear_layer_sz),
            nn.Linear(linear_layer_sz, output_layer_sz)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = [emb_layer(data=data, col_num=i) for i, emb_layer in enumerate(self.embed_layers)]
        x = torch.cat(x, 1)
        x = self.bnorm_emb(x)
        x = self.feed_fwd(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device >>', device)
    embedding_details = [(10, 5), (20, 8), (30, 12), (6, 3)]
    sample_data = torch.tensor(
        [[7, 13, 21, 3],
         [5, 17, 15, 1],
         [3, 12, 11, 4],
         [5, 15, 27, 1]]
    ).to(device)
    model = TorchTabular(embed_details=embedding_details,
                         dropouts=0.3,
                         linear_layer_sz=100,
                         output_layer_sz=1).to(device)
    out = model(data=sample_data)
    print('Output Shape >>', out.shape)
    print('Output >>', out)

