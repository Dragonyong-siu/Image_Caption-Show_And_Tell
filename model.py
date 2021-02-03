# decode : gru

import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.weight_norm import weight_norm
import random

device = 'cuda'
cnn = ptcv_get_model('resnet34', pretrained = True) # not trainable parameters

class decoder(nn.Module):
  def __init__(self, features_dim, gru_dim, embed_dim, seq_len = hyper_parameters['max_len'], dropout = 0.5):
    super(decoder, self).__init__()
    self.features_dim = features_dim
    self.gru_dim = gru_dim
    self.embed_dim = embed_dim
    self.seq_len = seq_len
    self.vocab_dim = hyper_parameters['vocab_dim']
    
    self.__encoder__ = cnn.features
    self.__img2embed_layer__ = nn.Linear(self.features_dim, self.embed_dim)
    self.__word2embed_layer__ = nn.Embedding(self.vocab_dim, self.embed_dim)
    self.__language_gru__ = nn.GRUCell(self.embed_dim, self.gru_dim)
    self.__fc_layer__ = nn.Linear(self.gru_dim, self.vocab_dim)

    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = dropout)
    self.dropout1 = nn.Dropout(p = 0.2)

    self.__init_weights__()

  def __init_weights__(self):
    self.__word2embed_layer__.weight.data.uniform_(-0.1, 0.1)
    self.__fc_layer__.bias.data.fill_(0)
    self.__fc_layer__.weight.data.uniform_(-0.1, 0.1)

  def __init_gru_state__(self, batch):
    h = torch.zeros(batch, self.gru_dim).to(device)

    return h

  def __random_topk__(self, pred, k): 
    prob_distribution = self.softmax(pred)
    top_indices = prob_distribution.topk(k = k).indices.permute(1, 0)

    return random.choice(top_indices)

  def forward(self, images, input_ids = None):
    batch = images.shape[0] # (N)
    batch_features = self.__encoder__(images).reshape(batch, self.features_dim) # (N, features_dim)
    imgs_embed = self.__img2embed_layer__(batch_features) # (N, embed_dim)
    
    h1 = self.__init_gru_state__(batch) # (N, gru_dim)

    preds = []
    for step in range(self.seq_len + 1):
      if step == 0:
        h1 = self.__language_gru__(self.dropout1(imgs_embed), h1)
        gru_input = self.__word2embed_layer__(torch.Tensor(batch * [1]).to(device).long())

      else:       
        h1 = self.__language_gru__(self.dropout1(gru_input), h1) 

        pred = self.__fc_layer__(self.dropout(h1)) # (N, vocab_dim)
        preds.append(pred.unsqueeze(1))

        if (input_ids is not None) & (step != self.seq_len): # train & valid
          gru_input = self.__word2embed_layer__(input_ids[:, step])
        else: # inference
          gru_input = self.__word2embed_layer__(self.__random_topk__(pred = pred, k =  1))

    return torch.cat(preds, dim = 1)

gru_decoder = decoder(features_dim = 512,
                      gru_dim = 1024,
                      embed_dim = 512)
