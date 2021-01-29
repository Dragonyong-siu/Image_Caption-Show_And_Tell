# set environment

!git clone https://github.com/poojahira/image-captioning-bottom-up-top-down

import sys
sys.path.insert(0, '/content/image-captioning-bottom-up-top-down/nlg-eval-master')

!pip install -U albumentations
!pip install pytorchcv
!pip install transformers

gpt2_tokenizer = 0.0
bert_tokenizer = 0.0

path = '/content/gdrive/My Drive/coco_image_caption/'

from google.colab import drive
drive.mount('/content/gdrive/', force_remount = True)
