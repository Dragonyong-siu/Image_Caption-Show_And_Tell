# Show-and-Tell-A-Neural-Image-Caption-Generator-pytorch

 Show and Tell : A Neutral Caption Generator

0) Download Coco_Dataset.zip and Unzip

1) Data_Tranforms 
 1.1) Resize to (256, 256, 3)
 1.2) Make couple :(Image, Caption_Target)

2) Image_Caption_Encoder
  CNN_Feature_Extraction(VGG_Net)(8 * 8 * 768)

3) Image_Caption_Dataset
 3.1) Image_Caption_Dataset_GPT2 
   Feature_Map from Image
   Encoded Target that will be captions : GPT2

4) Image_Caption_Decoder 
 4.1) Image_Caption_Decoder_LSTM
  
5) Image_Caption_Loss : CrossEntropyLoss

6) Image_Caption_Train

: I made 3 Models for Image Captioning
 1.LSTMCell_Decoder
 2.LSTM_Decoder
 3.GPT2_Decoder
 
