import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size=500002,
                        embed_size=300, 
                        hidden_size=50, 
                        num_classes=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        # self.rnn = nn.RNN(embed_size, hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size,batch_first=True)
        self.cls = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        # out, hidden = self.rnn(x)
        # print('shapes')
        # input to GRU should be [batch_size, seq_len, emb_size]
        # print(x.shape)
        output, hidden = self.rnn(x)
        # print(output.shape)
        # print(hidden.shape)
        # out = [batch_size, seq_len, n_classes]
        output = self.cls(hidden[0])
        return output