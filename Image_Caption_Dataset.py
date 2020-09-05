import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model

special_tokens = '[PAD]', '[START]', '[END]'
GPT2_Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
GPT2_Config = GPT2Config.from_pretrained('gpt2')
GPT2_Model = GPT2Model(GPT2_Config).from_pretrained('gpt2', config = GPT2_Config)
GPT2_Tokenizer.add_tokens(special_tokens)
GPT2_Model.resize_token_embeddings(len(GPT2_Tokenizer))

class Dataset_A(torch.utils.data.Dataset):
  def __init__(self, data, max_len, feature_extractor):
    self.data = data
    self.max_len = max_len
    self.feature_extractor = feature_extractor
    self.config = GPT2_Config
    self.model = GPT2_Model
    self.tokenizer = GPT2_Tokenizer
    self.Encoded_PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')
    self.Encoded_START = self.tokenizer.convert_tokens_to_ids('[START]')
    self.Encoded_END = self.tokenizer.convert_tokens_to_ids('[END]')
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}
    
    # Feature_Map
    PIL_Image = self.data[index][0]
    Image_Array = np.asarray(PIL_Image)
    Copied_Image_Array = Image_Array.copy()
    Image_Tensor = torch.Tensor(Copied_Image_Array)

    # Normalize
    Image_Tensor = Image_Tensor.view(3, 256, 256)
    Normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), 
                                                     (0.229, 0.224, 0.225))
    Image_Tensor = Normalization(Image_Tensor)
    Image_Tensor = Image_Tensor.unsqueeze(0)
    Image_Tensor = Image_Tensor.to(device)
    Feature_Map = self.feature_extractor(Image_Tensor)

    # Caption_Ids
    # Caption_Target
    Target_List = self.data[index][1]
    Target_List = [Target_List[0], Target_List[1]]
    Target_Caption = " ".join(Target_List).lower()
    Tokenized_Caption = self.tokenizer.tokenize(Target_Caption)
    Encoded_Caption = self.tokenizer.encode(Tokenized_Caption)
    
    
    if len(Encoded_Caption) >= (self.max_len - 2):
      Caption_Ids = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)]
      Caption_Target = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)] \
      + [self.Encoded_END]
    else:
      Caption_Ids = [self.Encoded_START] + Encoded_Caption
      Caption_Target = [self.Encoded_START] + Encoded_Caption + [self.Encoded_END]
  
    # Padding
    Padding_Length = self.max_len - len(Caption_Ids)
    Caption_Ids = Padding(Caption_Ids, self.Encoded_PAD, Padding_Length)
    Padding_Length = self.max_len - len(Caption_Target)
    Caption_Target = Padding(Caption_Target, self.Encoded_PAD, Padding_Length)

    # Rezist to Dictionary
    Dictionary['Target_List'] = Target_List
    Dictionary['Feature_Map'] = Feature_Map.squeeze(0)
    Dictionary['Caption_Ids'] = torch.Tensor(Caption_Ids)
    Dictionary['Caption_Target'] = torch.Tensor(Caption_Target)

    return Dictionary

class Dataset_B(torch.utils.data.Dataset):
  def __init__(self, data, max_len, feature_extractor):
    self.data = data
    self.max_len = max_len
    self.feature_extractor = feature_extractor
    self.config = GPT2_Config
    self.model = GPT2_Model
    self.tokenizer = GPT2_Tokenizer
    self.Encoded_PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')
    self.Encoded_START = self.tokenizer.convert_tokens_to_ids('[START]')
    self.Encoded_END = self.tokenizer.convert_tokens_to_ids('[END]')
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}
    
    # Feature_Map
    PIL_Image = self.data[index][0]
    Image_Array = np.asarray(PIL_Image)
    Copied_Image_Array = Image_Array.copy()
    Image_Tensor = torch.Tensor(Copied_Image_Array)

    # Normalize
    Image_Tensor = Image_Tensor.view(3, 256, 256)
    Normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), 
                                                     (0.229, 0.224, 0.225))
    Image_Tensor = Normalization(Image_Tensor)
    Image_Tensor = Image_Tensor.unsqueeze(0)
    Image_Tensor = Image_Tensor.to(device)
    Feature_Map = self.feature_extractor(Image_Tensor)

    # Caption_Ids
    # Caption_Target
    Target_List = self.data[index][1]
    Target_List = [Target_List[1], Target_List[2]]
    Target_Caption = " ".join(Target_List).lower()
    Tokenized_Caption = self.tokenizer.tokenize(Target_Caption)
    Encoded_Caption = self.tokenizer.encode(Tokenized_Caption)
    
    
    if len(Encoded_Caption) >= (self.max_len - 2):
      Caption_Ids = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)]
      Caption_Target = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)] \
      + [self.Encoded_END]
    else:
      Caption_Ids = [self.Encoded_START] + Encoded_Caption
      Caption_Target = [self.Encoded_START] + Encoded_Caption + [self.Encoded_END]
  
    # Padding
    Padding_Length = self.max_len - len(Caption_Ids)
    Caption_Ids = Padding(Caption_Ids, self.Encoded_PAD, Padding_Length)
    Padding_Length = self.max_len - len(Caption_Target)
    Caption_Target = Padding(Caption_Target, self.Encoded_PAD, Padding_Length)

    # Rezist to Dictionary
    Dictionary['Target_List'] = Target_List
    Dictionary['Feature_Map'] = Feature_Map.squeeze(0)
    Dictionary['Caption_Ids'] = torch.Tensor(Caption_Ids)
    Dictionary['Caption_Target'] = torch.Tensor(Caption_Target)

    return Dictionary

class Dataset_C(torch.utils.data.Dataset):
  def __init__(self, data, max_len, feature_extractor):
    self.data = data
    self.max_len = max_len
    self.feature_extractor = feature_extractor
    self.config = GPT2_Config
    self.model = GPT2_Model
    self.tokenizer = GPT2_Tokenizer
    self.Encoded_PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')
    self.Encoded_START = self.tokenizer.convert_tokens_to_ids('[START]')
    self.Encoded_END = self.tokenizer.convert_tokens_to_ids('[END]')
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}
    
    # Feature_Map
    PIL_Image = self.data[index][0]
    Image_Array = np.asarray(PIL_Image)
    Copied_Image_Array = Image_Array.copy()
    Image_Tensor = torch.Tensor(Copied_Image_Array)

    # Normalize
    Image_Tensor = Image_Tensor.view(3, 256, 256)
    Normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), 
                                                     (0.229, 0.224, 0.225))
    Image_Tensor = Normalization(Image_Tensor)
    Image_Tensor = Image_Tensor.unsqueeze(0)
    Image_Tensor = Image_Tensor.to(device)
    Feature_Map = self.feature_extractor(Image_Tensor)

    # Caption_Ids
    # Caption_Target
    Target_List = self.data[index][1]
    Target_List = [Target_List[2], Target_List[3]]
    Target_Caption = " ".join(Target_List).lower()
    Tokenized_Caption = self.tokenizer.tokenize(Target_Caption)
    Encoded_Caption = self.tokenizer.encode(Tokenized_Caption)
    
    
    if len(Encoded_Caption) >= (self.max_len - 2):
      Caption_Ids = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)]
      Caption_Target = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)] \
      + [self.Encoded_END]
    else:
      Caption_Ids = [self.Encoded_START] + Encoded_Caption
      Caption_Target = [self.Encoded_START] + Encoded_Caption + [self.Encoded_END]
  
    # Padding
    Padding_Length = self.max_len - len(Caption_Ids)
    Caption_Ids = Padding(Caption_Ids, self.Encoded_PAD, Padding_Length)
    Padding_Length = self.max_len - len(Caption_Target)
    Caption_Target = Padding(Caption_Target, self.Encoded_PAD, Padding_Length)

    # Rezist to Dictionary
    Dictionary['Target_List'] = Target_List
    Dictionary['Feature_Map'] = Feature_Map.squeeze(0)
    Dictionary['Caption_Ids'] = torch.Tensor(Caption_Ids)
    Dictionary['Caption_Target'] = torch.Tensor(Caption_Target)

    return Dictionary

class Dataset_D(torch.utils.data.Dataset):
  def __init__(self, data, max_len, feature_extractor):
    self.data = data
    self.max_len = max_len
    self.feature_extractor = feature_extractor
    self.config = GPT2_Config
    self.model = GPT2_Model
    self.tokenizer = GPT2_Tokenizer
    self.Encoded_PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')
    self.Encoded_START = self.tokenizer.convert_tokens_to_ids('[START]')
    self.Encoded_END = self.tokenizer.convert_tokens_to_ids('[END]')
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}
    
    # Feature_Map
    PIL_Image = self.data[index][0]
    Image_Array = np.asarray(PIL_Image)
    Copied_Image_Array = Image_Array.copy()
    Image_Tensor = torch.Tensor(Copied_Image_Array)

    # Normalize
    Image_Tensor = Image_Tensor.view(3, 256, 256)
    Normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), 
                                                     (0.229, 0.224, 0.225))
    Image_Tensor = Normalization(Image_Tensor)
    Image_Tensor = Image_Tensor.unsqueeze(0)
    Image_Tensor = Image_Tensor.to(device)
    Feature_Map = self.feature_extractor(Image_Tensor)

    
    # Caption_Ids
    # Caption_Target
    Target_List = self.data[index][1]
    Target_List = [Target_List[3], Target_List[4]]
    Target_Caption = " ".join(Target_List).lower()
    Tokenized_Caption = self.tokenizer.tokenize(Target_Caption)
    Encoded_Caption = self.tokenizer.encode(Tokenized_Caption)
    
    if len(Encoded_Caption) >= (self.max_len - 2):
      Caption_Ids = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)]
      Caption_Target = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)] \
      + [self.Encoded_END]
    else:
      Caption_Ids = [self.Encoded_START] + Encoded_Caption
      Caption_Target = [self.Encoded_START] + Encoded_Caption + [self.Encoded_END]
  
    # Padding
    Padding_Length = self.max_len - len(Caption_Ids)
    Caption_Ids = Padding(Caption_Ids, self.Encoded_PAD, Padding_Length)
    Padding_Length = self.max_len - len(Caption_Target)
    Caption_Target = Padding(Caption_Target, self.Encoded_PAD, Padding_Length)

    # Rezist to Dictionary
    Dictionary['Target_List'] = Target_List
    Dictionary['Feature_Map'] = Feature_Map.squeeze(0)
    Dictionary['Caption_Ids'] = torch.Tensor(Caption_Ids)
    Dictionary['Caption_Target'] = torch.Tensor(Caption_Target)

    return Dictionary

def Padding(X, padding_value, padding_length):
  return X + [padding_value] * padding_length


from torch.utils.data import DataLoader
data_input = Train_data
caption_length = 36
Image_Caption_Dataset_GPT2 = Dataset_A(data_input, caption_length, Image_Caption_Encoder) + \
                             Dataset_B(data_input, caption_length, Image_Caption_Encoder) + \
                             Dataset_C(data_input, caption_length, Image_Caption_Encoder) + \
                             Dataset_D(data_input, caption_length, Image_Caption_Encoder)
Train_dataloader = DataLoader(Image_Caption_Dataset_GPT2,
                              batch_size = 32,
                              shuffle = True,
                              drop_last = True)
