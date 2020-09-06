import random
class Image_Caption_Model_LSTM(nn.Module):
  def __init__(self):
    super(Image_Caption_Model_LSTM, self).__init__()
    self.GPT2_Model = GPT2_Model
    self.GPT2_wte = self.GPT2_Model.wte
    self.GPT2_Hidden = 768
    self.LSTM_Layers = 2
    self.LSTM_Model = nn.LSTM(self.GPT2_Hidden, self.GPT2_Hidden, self.LSTM_Layers)
    self.Linear_FC = nn.Linear(8 * 8 * 512, self.GPT2_Hidden)
    self.Linear_LM = nn.Linear(self.GPT2_Hidden, 50260)
    
    self.ReLU = nn.ReLU(inplace = True)
    self.Dropout = nn.Dropout(0.001, inplace = True)
    self.Sequence_length = 40
    self.Batch_size = 8

  def forward(self, input_ids, feature_map):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    feature_map = self.Linear_FC(feature_map)
   
    input_wte = self.GPT2_wte(input_ids)[:, :-1]
    input_Embedding = torch.cat([feature_map.unsqueeze(1), input_wte], dim = 1) 
    input_Embedding = input_Embedding.to(device) 
    input_Embedding = input_Embedding.transpose(1, 0) 

    hidden = torch.zeros((self.LSTM_Layers, self.Batch_size, self.GPT2_Hidden)).to(device)
    cell = torch.zeros((self.LSTM_Layers, self.Batch_size, self.GPT2_Hidden)).to(device)
    output, (h_n, c_n) = self.LSTM_Model(input_Embedding, (hidden, cell))
    
    Hidden_states = output.transpose(1, 0)
    Logits = self.Linear_LM(Hidden_states)
    Logits = self.Dropout(Logits)
    return Logits
  
  def Image_Caption_Sampling(self, feature_map, max_len):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    Sampling_inputs = self.Linear_FC(feature_map) 

    Sample_Ids = []
    hidden = torch.zeros((self.LSTM_Layers, 1, self.GPT2_Hidden)).to(device)
    cell = torch.zeros((self.LSTM_Layers, 1, self.GPT2_Hidden)).to(device)
    for i in range(max_len):
      Sampling_inputs = Sampling_inputs.unsqueeze(0)
      Sampling_inputs = Sampling_inputs.transpose(1, 0).to(device) 
      Sampling_outputs, (hidden, cell) = self.LSTM_Model(Sampling_inputs, (hidden, cell))

      Sampling_outputs = self.Linear_LM(Sampling_outputs.transpose(1, 0).squeeze(1))
      Words_Index = Next_Word_Index(Sampling_outputs)
      Words_Index = torch.Tensor([Words_Index]).long().to(device)
      Sample_Ids.append(Words_Index) 
      
      Sampling_inputs = self.GPT2_wte(Words_Index) 
      Sampling_inputs = Sampling_inputs.to(device)
    Sample_Ids = torch.stack(Sample_Ids, dim = 1)
    return Sample_Ids

def Next_Word_Index(logits):
  Last_Word_Embedding = logits[0, :]
  Softmax_logits = torch.softmax(Last_Word_Embedding, dim = 0)
  Words_Probability = Softmax_logits.tolist()
  Words_Sorted = sorted(Words_Probability)

  First_value = Words_Sorted[-1]
  Second_value = Words_Sorted[-2]
  Third_value = Words_Sorted[-3]
  Fourth_value = Words_Sorted[-4]

  First_Index = Words_Probability.index(First_value)
  Second_Index = Words_Probability.index(Second_value)
  Third_Index = Words_Probability.index(Third_value)
  Fourth_Index = Words_Probability.index(Fourth_value)

  Index_List = [(First_Index, First_value),
                (Second_Index, Second_value),
                (Third_Index, Third_value),
                (Fourth_Index, Fourth_value)]

  Index_MAZINO = []
  for i in range(len(Index_List)):
    if Index_List[i][1] >= 0.2:
      Index_MAZINO.append(Index_List[i][0])
  
  if len(Index_MAZINO) == 0:
    Index_MAZINO = [First_Index]

  Words_Index = random.choice(Index_MAZINO)
  return Words_Index 

Image_Caption_Decoder = Image_Caption_Model_LSTM().to(device)
