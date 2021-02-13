# model3 : encoder-vgg, decoder-transformer

import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.weight_norm import weight_norm

device = 'cuda'
cnn = ptcv_get_model('vgg19', pretrained = True)#'efficientnet_b7c', pretrained = True) # not trainable parameters

class decoder(nn.Module):
  def __init__(self, features_dim, embed_dim, seq_len = hyper_parameters['max_len'], dropout = 0.5):
    super(decoder, self).__init__()
    self.features_dim = features_dim
    self.embed_dim = embed_dim
    self.seq_len = seq_len
    self.vocab_dim = hyper_parameters['vocab_dim']
    self.layer_num = hyper_parameters['layer_num']
    self.block_num = hyper_parameters['block_num']
    self.block_size = hyper_parameters['block_size']

    self.__encoder__ = cnn.features
    self.__img2embed_layer__ = nn.Linear(self.features_dim, self.embed_dim)
    self.__content_embed__ = gpt2_model.wte
    self.__position_embed__ = gpt2_model.wpe
    self.__embed_drop__ = nn.Dropout(p = 0.5)
    self.__hidden_layers__ = gpt2_model.h
    self.__layer_norm__ = gpt2_model.ln_f
    self.__fc_layer__ = nn.Linear(self.embed_dim, self.vocab_dim)

    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = dropout)

    self.__init_weights__()

  def __init_weights__(self):
    ## self.__fc_layer__.bias.data.fill_(0)
    ## self.__fc_layer__.weight.data.uniform_(-0.1, 0.1)
    torch.nn.init.xavier_uniform_(self.__fc_layer__.weight)

  def __random_topk__(self, pred, k): 
    prob_distribution = self.softmax(pred)
    top_indices = prob_distribution.topk(k = k).indices.permute(1, 0)

    return random.choice(top_indices)

  def __img_block__(self, images, block_num):
    batch, c, h, w = images.shape

    blocks = []
    for y in range(0, h, int(h/block_num)):
      for x in range(0, w, int(w/block_num)):
        blocks.append(images[:, :, y:(y + int(h/block_num)), x:(x + int(w/block_num))])
    return torch.stack(blocks, dim = 1)

  def forward(self, images, input_ids = None):
    batch = images.shape[0] # (N)

    img_blocks = self.__img_block__(images, self.block_num).reshape(-1, 3, self.block_size, self.block_size)
    with torch.no_grad():
      batch_features = self.__encoder__(img_blocks).reshape(batch, int(self.block_num ** 2), self.features_dim) 
    imgs_embed = self.__img2embed_layer__(batch_features) # (N, block_num ** 2, embed_dim)

    words_embed = self.__content_embed__(input_ids)
    indices  = torch.arange(self.seq_len + int(self.block_num ** 2)).expand(batch, -1).to(device)
    position_embed = self.__position_embed__(indices)

    h = self.__embed_drop__(torch.cat([imgs_embed, words_embed], dim = 1) + position_embed).to(device) # (N, seq_len, embed_dim)
    for i in range(self.layer_num):
      h = self.__hidden_layers__[i](h)[0]
      h[:, :(self.block_num ** 2), :] = imgs_embed + position_embed[:, :(self.block_num ** 2), :]

    preds = self.__fc_layer__(self.dropout(self.__layer_norm__(h))) # (N, seg_len + block_num ** 2, vocab_dim)
    return preds[:, int(self.block_num ** 2):, :]
  
  def __sample__(self, images):
    batch = images.shape[0] # (N)
  
    img_blocks = self.__img_block__(images, self.block_num).reshape(-1, 3, self.block_size, self.block_size)
    batch_features = self.__encoder__(img_blocks).reshape(batch, int(self.block_num ** 2), self.features_dim)
    imgs_embed = self.__img2embed_layer__(batch_features) # (N, block_num ** 2, embed_dim)
     
    start_embed = self.__content_embed__(torch.Tensor(batch * [50258]).to(device).long())  
    indices = torch.arange(int(self.block_num ** 2) + 1).expand(batch, -1).to(device).long()
    position_embed = self.__position_embed__(indices)

    h = (torch.cat([imgs_embed, start_embed.unsqueeze(1)], dim = 1) + position_embed).to(device) # (N, block_num ** 2 + 1, embed_dim)

    preds = torch.zeros([batch, self.seq_len]).to(device)
    scores = torch.zeros([batch, self.seq_len, self.vocab_dim]).to(device)
    for i in range(self.seq_len):
      for j in range(self.layer_num):
        h = self.__hidden_layers__[j](h)[0]
        h[:, :(self.block_num ** 2), :] = imgs_embed + position_embed[:, :(self.block_num ** 2), :]

      pred = self.__fc_layer__(self.__layer_norm__(h))[:, -1, :] # (N, vocab_dim)
      preds[:, i] = self.__random_topk__(pred = pred, k = 1) 
      scores[:, i, :] = pred
      
      words_embed = self.__content_embed__(preds[:, :(i + 1)].long())
      indices = torch.arange(i + int(self.block_num ** 2) + 2).expand(batch, -1).to(device)
      position_embed = self.__position_embed__(indices)

      h = torch.cat([imgs_embed, start_embed.unsqueeze(1), words_embed], dim = 1) + position_embed

    return scores

gpt_decoder = decoder(features_dim = 512 * 2 * 2,#2560,
                      embed_dim = 768)
