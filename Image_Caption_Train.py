from tqdm import tqdm
def Image_Caption_Train(dataloader, model, optimizer, device):
  model.train()
  Book = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0
  for bi, Dictionary in enumerate(Book):
    Caption_Ids = Dictionary['Caption_Ids']
    Feature_Map = Dictionary['Feature_Map']
    Caption_Target = Dictionary['Caption_Target']

    Caption_Ids = Caption_Ids.to(device).long()
    Feature_Map = Feature_Map.to(device)
    Caption_Target = Caption_Target.to(device).long()

    model.zero_grad()
    Logits = model(Caption_Ids, Feature_Map)
    
    Logits = Logits.view(-1, 50260)
    Caption_Target = Caption_Target.view(-1)

    Caption_Loss = Image_Caption_Loss(Logits, Caption_Target)
    Caption_Loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += Caption_Loss

  Average_Caption_Loss = total_loss / len(dataloader)
  print(" Average_Caption_Loss: {0:.2f}".format(Average_Caption_Loss))

def FIT(Encoder, Decoder, Epochs, Learning_Rate):
  Params = list(Encoder.parameters()) + list(Decoder.parameters())
  optimizer = torch.optim.AdamW(Params, lr = Learning_Rate)
  for i in range(Epochs):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Image_Caption_Train(Train_dataloader, Decoder, optimizer, device)
  torch.save(Encoder, '/content/gdrive/My Drive/' + f'Image_Caption_Encoder')
  torch.save(Decoder, '/content/gdrive/My Drive/' + f'Image_Caption_Decoder')
    
FIT(Image_Caption_Decoder, Image_Caption_Decoder, Epochs = 3, Learning_Rate = 1e-3)
