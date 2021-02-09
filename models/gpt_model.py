# model2 : encoder-resnet, decoder-transformer

import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

cnn = ptcv_get_model('vgg19', pretrained = True) # not trainable parameters

class decoder(nn.Module):
  def __init__(self, features_dim, embed_dim, seq_len = hyper_parameters['max_len'], dropout = 0.5):
    super(decoder, self).__init__()
    self.features_dim = features_dim
    self.embed_dim = embed_dim
    self.seq_len = seq_len
    self.vocab_dim = hyper_parameters['vocab_dim']
    self.layer_num = hyper_parameters['layer_num']

    self.__encoder__ = cnn.features
    self.__img2embed_layer__ = nn.Linear(self.features_dim, self.embed_dim)
    self.__content_embed__ = gpt2_model.wte
    self.__position_embed__ = gpt2_model.wpe
    self.__embed_drop__ = gpt2_model.drop
    self.__hidden_layers__ = gpt2_model.h
    self.__layer_norm__ = gpt2_model.ln_f
    self.__fc_layer__ = nn.Linear(self.embed_dim, self.vocab_dim)

    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = dropout)

    self.__init_weights__()

  def __init_weights__(self):
    self.__fc_layer__.bias.data.fill_(0)
    self.__fc_layer__.weight.data.uniform_(-0.1, 0.1)

  def __random_topk__(self, pred, k): 
    prob_distribution = self.softmax(pred)
    top_indices = prob_distribution.topk(k = k).indices.permute(1, 0)

    return random.choice(top_indices)

  def forward(self, images, input_ids = None):
    batch = images.shape[0] # (N)
    batch_features = self.__encoder__(images).reshape(batch, self.features_dim) # (N, features_dim)
    imgs_embed = self.__img2embed_layer__(batch_features) # (N, embed_dim)

    words_embed = self.__content_embed__(input_ids[:, 1:])
    indices = torch.arange(self.seq_len).expand(batch, -1).to(device)
    position_embed = self.__position_embed__(indices)

    h = self.__embed_drop__(torch.cat([imgs_embed.unsqueeze(1), words_embed], dim = 1) + position_embed).to(device) # (N, seq_len, embed_dim)
    for i in range(self.layer_num):
      h = self.__hidden_layers__[i](h)[0]
  
    preds = self.__fc_layer__(self.dropout(self.__layer_norm__(h))) # (N, seg_len, vocab_dim)
    return preds  
  
  def __sample__(self, images):
    batch = images.shape[0] # (N)
    batch_features = self.__encoder__(images).reshape(batch, self.features_dim) # (N, features_dim)
    imgs_embed = self.__img2embed_layer__(batch_features) # (N, embed_dim)
     
    h = (imgs_embed + self.__position_embed__(torch.zeros(batch).to(device).long())).unsqueeze(1) # (N, embed_dim)

    preds = torch.zeros([batch, self.seq_len]).to(device)
    scores = torch.zeros([batch, self.seq_len, self.vocab_dim]).to(device)
    for i in range(self.seq_len):
      for j in range(self.layer_num):
        h = self.__hidden_layers__[j](h)[0]

      pred = self.__fc_layer__(self.__layer_norm__(h))[:, -1, :] # (N, vocab_dim)
      preds[:, i] = self.__random_topk__(pred = pred, k = 1) 
      scores[:, i, :] = pred
      
      words_embed = self.__content_embed__(preds[:, :(i+1)].long())
      indices = torch.arange(i+2).expand(batch, -1).to(device)
      position_embed = self.__position_embed__(indices)

      h = torch.cat([imgs_embed.unsqueeze(1), words_embed], dim = 1) + position_embed

    return scores

gpt_decoder = decoder(features_dim = 512,
                      embed_dim = 768)
