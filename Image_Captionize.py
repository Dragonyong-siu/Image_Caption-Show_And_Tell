def Image_Captionize(image, encoder, decoder, tokenizer):
  Image_Array = np.asarray(image)
  Copied_Image_Array = Image_Array.copy()
  Image_Tensor = torch.Tensor(Copied_Image_Array)

  Image_Tensor = Image_Tensor.view(3, 256, 256)
  Normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), 
                                                   (0.229, 0.224, 0.225))
  Image_Tensor = Normalization(Image_Tensor)
  Image_Tensor = Image_Tensor.unsqueeze(0)
  Image_Tensor = Image_Tensor.to(device)
  Feature_Map = encoder(Image_Tensor)

  decoder_inputs = Feature_Map.unsqueeze(0).to(device)
  outputs = decoder.Image_Caption_Sampling(decoder_inputs, 30)
  outputs = outputs.squeeze(0).tolist()
  samples = []
  for i in range(len(outputs)):
    samples.append(tokenizer.decode(outputs[i]))
  Caption = "".join(samples)
  return Caption
