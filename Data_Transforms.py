import torchvision
from torchvision import transforms
Data_Transforms = torchvision.transforms.Compose([transforms.Resize((256, 256)),
                                                  transforms.RandomHorizontalFlip(p = 0.5)
                                                  #transforms.Grayscale(num_output_channels = 1),
                                                  #transforms.ToTensor(),
                                                  #transforms.Normalize((0.485, 0.456, 0.406),
                                                  #                     (0.229, 0.224, 0.225))
                                                  ])
Train_data = torchvision.datasets.CocoCaptions(root = 'val2014/',
                                               annFile = '/content/gdrive/My Drive/captions_val2014.json',
                                               transform = Data_Transforms,
                                               target_transform = None,
                                               transforms = None)
