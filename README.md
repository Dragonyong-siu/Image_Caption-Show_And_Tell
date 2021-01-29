# Show-and-Tell-A-Neural-Image-Caption-Generator-pytorch



## getting started
This repository is a pytorch implementation of [Show-and-Tell-A-Neural-Image-Caption-Generator](https://arxiv.org/pdf/1411.4555v2.pdf) for Image Captioning. 













## encoder : resnet50
I used resnet50 as an encoder. Larger cnn models such as efficientnet and resnet152 could have been used, but in this project, encoders were not included in the trainable params, so I chose smaller models.











## coco_dataset: data prepare
The coco data consists of 80k train images, 40k valid images, and 40k test images.
Here, I did not use test data, but trained on 80k images, and only did validation on 40k images.

download images here : ['train_coco_images2014'](http://images.cocodataset.org/zips/train2014.zip),
['valid_coco_images2014'](http://images.cocodataset.org/zips/val2014.zip), ['test_coco_images2014'](http://images.cocodataset.org/zips/test2014.zip)


download  caption annotation here : 
(http://images.cocodataset.org/annotations/annotations_trainval2014.zip)











