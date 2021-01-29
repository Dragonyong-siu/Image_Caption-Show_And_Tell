# Show-and-Tell-A-Neural-Image-Caption-Generator-pytorch



## getting started
This repository is a pytorch implementation of [Show-and-Tell-A-Neural-Image-Caption-Generator](https://arxiv.org/pdf/1411.4555v2.pdf) for Image Captioning. 













## coco_dataset: data prepare
The coco data consists of 80k train images, 40k valid images, and 40k test images.
Here, I did not use test data, but trained on 80k images, and only did validation on 40k images.

download images here : ['train_coco_images2014'](http://images.cocodataset.org/zips/train2014.zip),
['valid_coco_images2014'](http://images.cocodataset.org/zips/val2014.zip), ['test_coco_images2014'](http://images.cocodataset.org/zips/test2014.zip)


download  caption annotation here : 
(http://images.cocodataset.org/annotations/annotations_trainval2014.zip)











## vocab
As a vocabulary for embeddedding. I tried using gpt2 (50257 tokens) and Bert (30232 tokens), but this required a relatively large amount of computation and was slow at learning, so I created vocab_dict separately.(See vocab.py for this.)

I selected frequently used words from the coco annotation data and proceeded with encoding.(I selected 5,000 tokens.)












## encoder : resnet50
I used resnet50 as an encoder. Larger cnn models such as efficientnet and resnet152 could have been used, but in this project, encoders were not included in the trainable params, so I chose smaller models.












## decoder : gru_cell
The decoder structure of the show and tell is built using rnn as a basic structure, which is called the beginning of image caption.

I wrote the decoder code using lstm, gru, and basic rnn according to the content of the paper, and among them, gru obtained the fastest convergence and the highest score.






-





## evaluation
