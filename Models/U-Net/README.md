# Multi-Class Image Segmentation of Aerial Images:
## Using U-Net



* **Training:** [![image](https://colab.research.google.com/assets/colab-badge.svg)]()
* **Testing:** [![image](https://colab.research.google.com/assets/colab-badge.svg)]()

This notebook is used to generate multi-class masks of Aerial/Satellite Images using Transfer Learning Technique, utilizing U-Net as the architechure of the Deep-Learning Model

Make sure you use GPU runtime for this notebook. For Google Colab, go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator. As using only `CPU` runtime will be very slow for training model.



## Table of Contents

- [Multi-Class Image Segmentation of Aerial Images](#multi-class-image-segmentation-of-aerial-images)
- [Table of Contents](#table-of-contents)
- [U-Net](#u-net)
- [UNET - Network Architecture](#unet---network-architecture)
- [Errors](#errors)
- [Prediction with Smooth Blending](#prediction-with-smooth-blending)
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
10 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![10 epoch: predict without smooth blending](./assets/prediction_10_epochs.png)     | ![10 epoch: predict with smooth blending](./assets/smoothprediction_10_epochs.png)
20 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![20 epoch: predict without smooth blending](./assets/prediction_20_epochs.png)     | ![20 epoch: predict with smooth blending](./assets/smoothprediction_20_epochs.png)
30 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![30 epoch: predict without smooth blending](./assets/prediction_30_epochs.png)     | ![30 epoch: predict with smooth blending](./assets/smoothprediction_30_epochs.png)
40 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![40 epoch: predict without smooth blending](./assets/prediction_40_epochs.png)     | ![40 epoch: predict with smooth blending](./assets/smoothprediction_40_epochs.png)
50 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![50 epoch: predict without smooth blending](./assets/prediction_50_epochs.png)     | ![50 epoch: predict with smooth blending](./assets/smoothprediction_50_epochs.png)
60 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![60 epoch: predict without smooth blending](./assets/prediction_60_epochs.png)     | ![60 epoch: predict with smooth blending](./assets/smoothprediction_60_epochs.png)
70 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![70 epoch: predict without smooth blending](./assets/prediction_70_epochs.png)     | ![70 epoch: predict with smooth blending](./assets/smoothprediction_70_epochs.png)
80 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![80 epoch: predict without smooth blending](./assets/prediction_80_epochs.png)     | ![80 epoch: predict with smooth blending](./assets/smoothprediction_80_epochs.png)
90 Epochs     | ![ground truth mask](./assets/ground_truth_mask.png) | ![90 epoch: predict without smooth blending](./assets/prediction_90_epochs.png)     | ![90 epoch: predict with smooth blending](./assets/smoothprediction_90_epochs.png)
100 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![100 epoch: predict without smooth blending](./assets/prediction_100_epochs.png)   | ![100 epoch: predict with smooth blending](./assets/smoothprediction_100_epochs.png)
110 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![110 epoch: predict without smooth blending](./assets/prediction_110_epochs.png)   | ![110 epoch: predict with smooth blending](./assets/smoothprediction_110_epochs.png)
120 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![120 epoch: predict without smooth blending](./assets/prediction_120_epochs.png)   | ![120 epoch: predict with smooth blending](./assets/smoothprediction_120_epochs.png)
130 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![130 epoch: predict without smooth blending](./assets/prediction_130_epochs.png)   | ![130 epoch: predict with smooth blending](./assets/smoothprediction_130_epochs.png)
140 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![140 epoch: predict without smooth blending](./assets/prediction_140_epochs.png)   | ![140 epoch: predict with smooth blending](./assets/smoothprediction_140_epochs.png)
150 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![150 epoch: predict without smooth blending](./assets/prediction_150_epochs.png)   | ![150 epoch: predict with smooth blending](./assets/smoothprediction_150_epochs.png)
160 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![160 epoch: predict without smooth blending](./assets/prediction_160_epochs.png)   | ![160 epoch: predict with smooth blending](./assets/smoothprediction_160_epochs.png)
170 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![170 epoch: predict without smooth blending](./assets/prediction_170_epochs.png)   | ![170 epoch: predict with smooth blending](./assets/smoothprediction_170_epochs.png)
180 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![180 epoch: predict without smooth blending](./assets/prediction_180_epochs.png)   | ![180 epoch: predict with smooth blending](./assets/smoothprediction_180_epochs.png)
190 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![190 epoch: predict without smooth blending](./assets/prediction_190_epochs.png)   | ![190 epoch: predict with smooth blending](./assets/smoothprediction_190_epochs.png)
200 Epochs    | ![ground truth mask](./assets/ground_truth_mask.png) | ![200 epoch: predict without smooth blending](./assets/prediction_200_epochs.png)   | ![200 epoch: predict with smooth blending](./assets/smoothprediction_200_epochs.png)



## Training Metrics

No. of Epoch | Loss   | Accuracy | Jaccard Coefficient | Validation Loss | Validation Accuracy | Validation Jaccard Coefficient 
:-----------:|:------:|:--------:|:-------------------:|:---------------:|:-------------------:|:------------------------------:
10           | 0.9504 | 0.7307   | 0.4767              | 0.9749          | 0.6644              | 0.3965
20           | 0.9263 | 0.8019   | 0.5960              | 0.9802          | 0.6807              | 0.4974
30           | 0.9152 | 0.8331   | 0.6464              | 0.9395          | 0.7677              | 0.6057
40           | 0.9061 | 0.8577   | 0.6873              | 0.9351          | 0.7916              | 0.6408
50           | 0.9039 | 0.8566   | 0.6968              | 0.9266          | 0.8124              | 0.6594
60           | 0.8964 | 0.8607   | 0.7068              | 0.9298          | 0.7972              | 0.6464
70           | 0.8849 | 0.8824   | 0.7476              | 0.9177          | 0.8356              | 0.6940
80           | 0.8769 | 0.8972   | 0.7743              | 0.9253          | 0.8187              | 0.6889
90           | 0.8751 | 0.9001   | 0.7794              | 0.9205          | 0.8237              | 0.6930
100          | 0.8748 | 0.9012   | 0.7831              | 0.9456          | 0.7970              | 0.6562
110          | 0.8692 | 0.9142   | 0.8067              | 0.9162          | 0.8466              | 0.7352
120          | 0.8650 | 0.9247   | 0.8320              | 0.9204          | 0.8391              | 0.7280
130          | 0.8644 | 0.9257   | 0.8338              | 0.9185          | 0.8419              | 0.7284
140          | 0.8625 | 0.9302   | 0.8442              | 0.9133          | 0.8522              | 0.7407
150          | 0.8608 | 0.9318   | 0.8472              | 0.9180          | 0.8515              | 0.7443
160          | 0.8588 | 0.9352   | 0.8537              | 0.9184          | 0.8467              | 0.7413
170          | 0.8564 | 0.9400   | 0.8643              | 0.9165          | 0.8532              | 0.7505
180          | 0.8561 | 0.9404   | 0.8650              | 0.9163          | 0.8544              | 0.7467
190          | 0.8546 | 0.9426   | 0.8707              | 0.9154          | 0.8577              | 0.7560
200          | 0.8554 | 0.9407   | 0.8656              | 0.9174          | 0.8531              | 0.7465



## Video of Progress Over Epochs
* [Video of Progress Over Epochs](https://youtu.be/HQUXmCZ0_JI)



## References

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [drone-images-semantic-segmentation](https://github.com/ayushdabra/drone-images-semantic-segmentation) [![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J-PQgIJWOCb7hoc4eiOCna1gUf0cnMCW)
* [Dataset: Semantic segmentation of aerial imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
* [DigitalSreeni: 228 - Semantic segmentation of aerial (satellite) imagery using U-net](https://www.youtube.com/watch?v=jvZm8REF2KY)
* [Dr. Sreenivas Bhattiprolu: GitHub - 228_semantic_segmentation_of_aerial_imagery_using_unet](https://github.com/bnsreenu/python_for_microscopists/tree/master/228_semantic_segmentation_of_aerial_imagery_using_unet)
* [DigitalSreeni: 229 - Smooth blending of patches for semantic segmentation of large images (using U-Net)](https://www.youtube.com/watch?v=HrGn4uFrMOM&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=30)
* [Dr. Sreenivas Bhattiprolu: GitHub - 229_smooth_predictions_by_blending_patches](https://github.com/bnsreenu/python_for_microscopists/tree/master/229_smooth_predictions_by_blending_patches)
* [DigitalSreeni: Python tips and tricks - 8: Working with RGB (and Hex) masks for semantic segmentation](https://www.youtube.com/watch?v=sGAwx4GMe4E)
* [U-net for image segmentation](https://www.youtube.com/playlist?list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE)
* [Vooban: GitHub - Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)
* [Kaggle API](https://www.kaggle.com/docs/api)
* [Medium: What is UNET?](https://medium.com/analytics-vidhya/what-is-unet-157314c87634)