import random
class Image_Caption_Model_GPT2(nn.Module):
  def __init__(self):
    super(Image_Caption_Model_GPT2, self).__init__()
    self.GPT2_Model = GPT2_Model
    self.GPT2_wte = self.GPT2_Model.wte
    self.GPT2_wpe = self.GPT2_Model.wpe
    self.GPT2_drop = self.GPT2_Model.drop
    self.GPT2_h = self.GPT2_Model.h
    self.GPT2_ln_f = self.GPT2_Model.ln_f
    self.GPT2_Hidden = 768
    self.GPT2_layers = 12
    self.Linear_FC = nn.Linear(8 * 8 * 512, self.GPT2_Hidden)
    self.Linear_LM = nn.Linear(self.GPT2_Hidden, 50260)
    self.ReLU = nn.ReLU(inplace = True)
    self.Dropout = nn.Dropout(0.1, inplace = True)

  def forward(self, input_ids, position_ids, feature_map):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    feature_map = self.Linear_FC(feature_map)

    input_wte = self.GPT2_wte(input_ids)[:, :-1]
    input_Embedding = torch.cat([feature_map.unsqueeze(1), input_wte], dim = 1) 
    position_wpe = self.GPT2_wpe(position_ids) 
    
    input_Embedding = input_Embedding + position_wpe
    input_Embedding = self.GPT2_drop(input_Embedding)
    input_Embedding = input_Embedding.to(device) 
    
    for i in range(self.GPT2_layers):
      input_Embedding = self.GPT2_h[i](input_Embedding)[0]

    Hidden_states = self.GPT2_ln_f(input_Embedding)
    Logits = self.Linear_LM(Hidden_states)
    Logits = self.Dropout(Logits)
    return Logits
  
  def Image_Caption_Sampling(self, feature_map, max_len):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    feature_inputs = self.Linear_FC(feature_map)
    feature_inputs = feature_inputs.unsqueeze(1)

    Sample_Ids = []
    for i in range(max_len):
      Addition_inputs = torch.Tensor(Sample_Ids)
      Addition_inputs = Addition_inputs.long().to(device)
      Addition_inputs = self.GPT2_wte(Addition_inputs)
      Addition_inputs = Addition_inputs.unsqueeze(0)
     
      Sampling_inputs = torch.cat((feature_inputs, Addition_inputs), dim = 1)
      inputs_length = Sampling_inputs.size(1)
      Position_inputs = torch.Tensor(np.arange(inputs_length))
      Position_inputs = Position_inputs.long().to(device)
      Position_inputs = self.GPT2_wpe(Position_inputs)
      
      Model_inputs = Sampling_inputs + Position_inputs
      Model_inputs = Model_inputs.to(device) 
      Sampling_inputs = self.GPT2_drop(Sampling_inputs)
      for j in range(self.GPT2_layers):
        Model_inputs = self.GPT2_h[j](Model_inputs)[0] 
      
      hidden_states = self.GPT2_ln_f(Model_inputs)
      Sampling_outputs = self.Linear_LM(hidden_states)
      Words_Index = Next_Word_Index(Sampling_outputs[:, -1, :])
      Words_Index = torch.Tensor([Words_Index]).long().to(device)
      Sample_Ids.append(Words_Index) 
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
    if Index_List[i][1] >= 0.35:
      Index_MAZINO.append(Index_List[i][0])
  
  if len(Index_MAZINO) == 0:
    Index_MAZINO = [First_Index]

  Words_Index = random.choice(Index_MAZINO)
  return Words_Index 

Image_Caption_Decoder = Image_Caption_Model_GPT2().to(device)
