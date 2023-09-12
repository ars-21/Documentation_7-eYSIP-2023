# 1024-Batch-Size: Potsdam: Multi-Class Image Segmentation of Aerial Images using U-Net


* **Training:** [![image](https://colab.research.google.com/assets/colab-badge.svg)]()
* **Testing:** [![image](https://colab.research.google.com/assets/colab-badge.svg)]()
* **Resize using Interpolation:** [![image](https://colab.research.google.com/assets/colab-badge.svg)]()

This notebook is used to generate multi-class masks of Aerial/Satellite Images using Transfer Learning Technique, utilizing Standard U-Net Model (with 2 additional contraction and expansion layers) as the architechure of the Deep-Learning Model.

* Batch Size of **1024** Pixels

Make sure you use GPU runtime for this notebook. For Google Colab, go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator. As using only `CPU` runtime will be very slow for training model.



## Table of Contents

- [1024-Batch-Size: Potsdam: Multi-Class Image Segmentation of Aerial Images using U-Net](#1024-batch-size-potsdam-multi-class-image-segmentation-of-aerial-images-using-u-net)
- [Table of Contents](#table-of-contents)
- [U-Net](#u-net)
- [UNET - Network Architecture](#unet---network-architecture)
- [Errors](#errors)
- [Prediction with Smooth Blending](#prediction-with-smooth-blending)
- [Dataset](#dataset)
- [Resize using Interpolation](#resize-using-interpolation)
- [Training Curves](#training-curves)
- [Output](#output)
- [Training Metrics](#training-metrics)
- [Video of Progress Over Epochs](#video-of-progress-over-epochs)
- [References](#references)



## U-Net
[U-Net](https://arxiv.org/abs/1505.04597) was developed by Olaf Ronneberger et al. for Bio Medical Image Segmentation. The architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. Thus, it is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.

In the original paper, the UNet is described as follows:

![UNet](./assets/u-net-architecture.png)

<center><em>U-Net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.</em></center>



## UNET - Network Architecture
UNET is a U-shaped encoder-decoder network architecture, which consists of four encoder blocks and four decoder blocks that are connected via a bridge. The encoder network (contracting path) half the spatial dimensions and double the number of filters (feature channels) at each encoder block. Likewise, the decoder network doubles the spatial dimensions and half the number of feature channels.

* **Encoder Network**
The encoder network acts as the feature extractor and learns an abstract representation of the input image through a sequence of the encoder blocks. Each encoder block consists of two 3x3 convolutions, where each convolution is followed by a ReLU (Rectified Linear Unit) activation function. The ReLU activation function introduces non-linearity into the network, which helps in the better generalization of the training data. The output of the ReLU acts as a skip connection for the corresponding decoder block. Next, follows a 2x2 max-pooling, where the spatial dimensions (height and width) of the feature maps are reduced by half. This reduces the computational cost by decreasing the number of trainable parameters.
* **Skip Connections**
These skip connections provide additional information that helps the decoder to generate better semantic features. They also act as a shortcut connection that helps the indirect flow of gradients to the earlier layers without any degradation. In simple terms, we can say that skip connection helps in better flow of gradient while backpropagation, which in turn helps the network to learn better representation.
* **Bridge**
The bridge connects the encoder and the decoder network and completes the flow of information. It consists of two 3x3 convolutions, where each convolution is followed by a ReLU activation function.
* **Decoder Network**
The decoder network is used to take the abstract representation and generate a semantic segmentation mask. The decoder block starts with a 2x2 transpose convolution. Next, it is concatenated with the corresponding skip connection feature map from the encoder block. These skip connections provide features from earlier layers that are sometimes lost due to the depth of the network. After that, two 3x3 convolutions are used, where each convolution is followed by a ReLU activation function. The output of the last decoder passes through a 1x1 convolution with sigmoid activation. The sigmoid activation function gives the segmentation mask representing the pixel-wise classification.



## Errors

If the following error occurs:

![AttributeError: module 'keras.utils.generic_utils' has no attribute 'get_custom_objects'](./assets/Error.png)

Go to File: `/usr/local/lib/python3.10/dist-packages/efficientnet/keras.py` and change the following lines

* `from . import inject_keras_modules, init_keras_custom_objects` ---> `from . import inject_keras_modules, init_tfkeras_custom_objects`
* `init_keras_custom_objects()` ---> `init_tfkeras_custom_objects()`



## Prediction with Smooth Blending

* Make smooth predictions by blending image patches, such as for image segmentation, rather than jagged ones. 
* One challenge of using a U-Net for image segmentation is to have smooth predictions, especially if the receptive field of the neural network is a small amount of pixels.

![Example](./assets/example.gif)



## Dataset
Dataset from International Society for Photogrammetry and Remote Sensing (ISPRS) ISPRS Test Project on Urban Classification, 3D Building Reconstruction and Semantic Labeling.
The dataset consists of 38 patches (of the same size) of satellite imagery of the city: Potsdam, Germany.

Classes:
* Impervious surfaces (white)
* Buildings (blue)
* Low vegetation (cyan)
* Trees (green)
* Cars (yellow)
* Clutter (red)



## Resize using Interpolation

* Utilized to resize down large Aerial/Satellite Images dataset for training on local system/cloud service with limited RAM/GPU Memory
* INTER_LANCZOS4 (Lanczos interpolation): utilized for resizing down Satellite/Aerial Images for its ability to produce high-quality interpolation results. It can preserve fine details and reduce artifacts such as aliasing or blurring that can occur with other interpolation methods.
* INTER_NEAREST_EXACT (nearest neighbor interpolation): utilized for resizing down Ground Truth Masks to prevent erroneous values in mask apart from specific Class Colours

INTER_LANCZOS4                                 | INTER_NEAREST_EXACT
:---------------------------------------------:|:--------------------------------------------------------:
![INTER_LANCZOS4](./assets/INTER_LANCZOS4.png) | ![INTER_NEAREST_EXACT](./assets/INTER_NEAREST_EXACT.png)



## Training Curves

### Training and Validation Loss
![Training and Validation Loss Curve](./assets/validation_loss.png)

### Training and Validation IOU
![Training and Validation IOU Curve](./assets/validation_iou.png)



## Output:

Original Input Image                        |  Ground Truth Multi-Class Segmented Mask
:------------------------------------------:|:-------------------------------------------:
![input image](./assets/original_image.jpg) | ![ground truth mask](./assets/ground_truth_mask.png)

No. of Epochs |Ground Truth Mask                                     |  Predicted Mask without Smooth Blending                                             | Predicted Mask with Smooth Blending
:------------:|:----------------------------------------------------:|:-----------------------------------------------------------------------------------:|-------------------------------------------------------------------------------------:
25 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![25 epoch: predict without smooth blending](./assets/prediction_25_epochs.png)     | ![25 epoch: predict with smooth blending](./assets/smoothprediction_25_epochs.png)
50 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![50 epoch: predict without smooth blending](./assets/prediction_50_epochs.png)     | ![50 epoch: predict with smooth blending](./assets/smoothprediction_50_epochs.png)
75 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![75 epoch: predict without smooth blending](./assets/prediction_75_epochs.png)     | ![75 epoch: predict with smooth blending](./assets/smoothprediction_75_epochs.png)
100 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![100 epoch: predict without smooth blending](./assets/prediction_100_epochs.png)   | ![100 epoch: predict with smooth blending](./assets/smoothprediction_100_epochs.png)
125 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![125 epoch: predict without smooth blending](./assets/prediction_125_epochs.png)   | ![125 epoch: predict with smooth blending](./assets/smoothprediction_125_epochs.png)
150 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![150 epoch: predict without smooth blending](./assets/prediction_150_epochs.png)   | ![150 epoch: predict with smooth blending](./assets/smoothprediction_150_epochs.png)
175 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![175 epoch: predict without smooth blending](./assets/prediction_175_epochs.png)   | ![175 epoch: predict with smooth blending](./assets/smoothprediction_175_epochs.png)
200 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![200 epoch: predict without smooth blending](./assets/prediction_200_epochs.png)   | ![200 epoch: predict with smooth blending](./assets/smoothprediction_200_epochs.png)
225 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![225 epoch: predict without smooth blending](./assets/prediction_225_epochs.png)   | ![225 epoch: predict with smooth blending](./assets/smoothprediction_225_epochs.png)
250 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![250 epoch: predict without smooth blending](./assets/prediction_250_epochs.png)   | ![250 epoch: predict with smooth blending](./assets/smoothprediction_250_epochs.png)
275 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![275 epoch: predict without smooth blending](./assets/prediction_275_epochs.png)   | ![275 epoch: predict with smooth blending](./assets/smoothprediction_275_epochs.png)
300 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![300 epoch: predict without smooth blending](./assets/prediction_300_epochs.png)   | ![300 epoch: predict with smooth blending](./assets/smoothprediction_300_epochs.png)
325 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![325 epoch: predict without smooth blending](./assets/prediction_325_epochs.png)   | ![325 epoch: predict with smooth blending](./assets/smoothprediction_325_epochs.png)
350 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![350 epoch: predict without smooth blending](./assets/prediction_350_epochs.png)   | ![350 epoch: predict with smooth blending](./assets/smoothprediction_350_epochs.png)
375 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![375 epoch: predict without smooth blending](./assets/prediction_375_epochs.png)   | ![375 epoch: predict with smooth blending](./assets/smoothprediction_375_epochs.png)
400 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![400 epoch: predict without smooth blending](./assets/prediction_400_epochs.png)   | ![400 epoch: predict with smooth blending](./assets/smoothprediction_400_epochs.png)



## Training Metrics

No. of Epoch | Loss   | Accuracy | Jaccard Coefficient | Validation Loss | Validation Accuracy | Validation Jaccard Coefficient 
:-----------:|:------:|:--------:|:-------------------:|:---------------:|:-------------------:|:------------------------------:
25           | 0.9982 | 0.4222   | 0.1965              | 1.0288          | 0.2797              | 0.1372
50           | 0.9676 | 0.5967   | 0.3372              | 1.0221          | 0.3729              | 0.1915
75           | 0.9829 | 0.4875   | 0.2487              | 1.0269          | 0.2709              | 0.1441
100          | 0.9546 | 0.6430   | 0.3907              | 1.0247          | 0.4432              | 0.2487
125          | 0.9433 | 0.6947   | 0.4474              | 0.9948          | 0.5499              | 0.3280
150          | 0.9306 | 0.7092   | 0.4716              | 0.9741          | 0.6082              | 0.3846
175          | 0.9255 | 0.7268   | 0.4978              | 0.9656          | 0.6383              | 0.4156
200          | 0.9155 | 0.7504   | 0.5316              | 0.9609          | 0.6520              | 0.4352
225          | 0.9108 | 0.7643   | 0.5494              | 0.9573          | 0.6668              | 0.4524
250          | 0.9094 | 0.7676   | 0.5572              | 0.9622          | 0.6620              | 0.4541
275          | 0.9103 | 0.7661   | 0.5480              | 0.9482          | 0.6814              | 0.4665
300          | 0.9049 | 0.7841   | 0.5764              | 0.9523          | 0.6765              | 0.4691
325          | 0.9024 | 0.7919   | 0.5905              | 0.9564          | 0.6671              | 0.4630
350          | 0.8999 | 0.7990   | 0.6002              | 0.9507          | 0.6791              | 0.4735
375          | 0.8975 | 0.8051   | 0.6106              | 0.9435          | 0.7067              | 0.5079
400          | 0.8962 | 0.8091   | 0.6173              | 0.9636          | 0.6607              | 0.4605



## Video of Progress Over Epochs
* [Video of Progress Over Epochs](https://youtu.be/PFJXy5dHs_M)



## References

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [drone-images-semantic-segmentation](https://github.com/ayushdabra/drone-images-semantic-segmentation) [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J-PQgIJWOCb7hoc4eiOCna1gUf0cnMCW)
* [Dataset: Potsdam - ISPRS Test Project on Urban Classification, 3D Building Reconstruction and Semantic Labeling](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
* [DigitalSreeni: 228 - Semantic segmentation of aerial (satellite) imagery using U-net](https://www.youtube.com/watch?v=jvZm8REF2KY)
* [Dr. Sreenivas Bhattiprolu: GitHub - 228_semantic_segmentation_of_aerial_imagery_using_unet](https://github.com/bnsreenu/python_for_microscopists/tree/master/228_semantic_segmentation_of_aerial_imagery_using_unet)
* [DigitalSreeni: 229 - Smooth blending of patches for semantic segmentation of large images (using U-Net)](https://www.youtube.com/watch?v=HrGn4uFrMOM&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=30)
* [Dr. Sreenivas Bhattiprolu: GitHub - 229_smooth_predictions_by_blending_patches](https://github.com/bnsreenu/python_for_microscopists/tree/master/229_smooth_predictions_by_blending_patches)
* [DigitalSreeni: Python tips and tricks - 8: Working with RGB (and Hex) masks for semantic segmentation](https://www.youtube.com/watch?v=sGAwx4GMe4E)
* [U-net for image segmentation](https://www.youtube.com/playlist?list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE)
* [Vooban: GitHub - Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)
* [Medium: What is UNET?](https://medium.com/analytics-vidhya/what-is-unet-157314c87634)
* [Kaggle: potsdam-vaihingen](https://www.kaggle.com/datasets/bkfateam/potsdamvaihingen)
* [Papers With Code: ISPRS Potsdam (2D Semantic Labeling Contest - Potsdam)](https://paperswithcode.com/dataset/isprs-potsdam)
