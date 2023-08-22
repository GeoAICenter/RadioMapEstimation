# Two-Stage Radio Map Estimation

This repository builds on [previous work](https://github.com/GeoAICenter/radio-map-estimation-pimrc2023) in radio map estimation (RME), predicting radio power across a two-dimensional area. As in previous experiments, we don't have knowledge of transmitter location(s), and the environment contains buildings which block and reflect radio signals. We take as input a small number of radio power measurements and a mask showing the locations of these sampled measurements and the buildings in the environment. The model then outputs a complete radio map showing predicted radio power at all locations in the 2D area (optionally including predictions at building locations).

![Image](images/example_input_output.png)

*Sample Input and Output: Sampled Map and Environment Mask are fed as input to the model, which outputs Complete Radio Map. Input and ground truth maps are adapted from the RadioMapSeer Dataset, discussed below.*

The idea of two-stage radio map estimation is to split the task of RME into two parts: first, taking the sparsely sampled measurements and predicting a dense radio map *without reference to buildings*, as if the signal were propagating through empty space; second, taking this dense radio map prediction and convolving it with a map of building locations ("Building Mask") and sampled locations (not shown) to produce a more accurate radio map that takes the physical environment into account.

![Image](images/example_input_output_2.png)

*Sample Input and Output of Two-Stage RME: Instead of jumping straight from the sampled map to the complete radio map with buildings, an intermediate map is produced ("Dense Map") that ignores buildings and just extrapolates from sample strength and location.*

The intuition behind this is that radio propagation has unique physical characteristics, in particular a source of propagation (i.e. a transmitter) and a relationship between signal strength and distance from that source, that most image data don't have. A model that can learn these physical characteristics will presumably perform better than one that treats the radio map like a normal image. The first stage of the model is designed to learn these spatial relationships, and the second stage to combine this with environmental information. Our main model, MAE_CBAM, uses a masked vision transformer with learnable position encodings, self-attention, and cross-attention to accomplish the first goal, and a CNN / UNet with CBAM-style gated attention for the second. Other models replace / remove different elements of this main model to carry out ablation studies.

## Model Architectures

The two-stage RME models are composed of two sub-models, one for each stage. As stated above, for the main [MAE_CBAM](models/mae_cbam.py) model, these are a masked vision transformer and a UNet with CBAM gated attention. The masked vision transformer is adapted from the code for Masked Autoencoders Are Scalable Vision Learners ([paper](https://arxiv.org/abs/2111.06377) | [github](https://github.com/facebookresearch/mae)), and the UNet is adapted from our previous code with an added implemention of CBAM: Convolutional Block Attention Module ([paper](https://arxiv.org/abs/1807.06521) | [github](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/README_EN.md#6-cbam-attention-usage)).

For ablation tests, we also have an [MAE_UNet](models/mae_unet.py) model, which removes CBAM attention from the UNet in the second stage of the model; the [Interpolation_CBAM](models/interpolation_cbam.py) and [Interpolation_UNet](models/interpolation_unet.py) models, which replace the masked vision transformer with basic interpolation; and the [CBAM](models/cbam.py) and [UNet](models/unet.py) models which remove the first stage altogether and feed the sparsely sampled map directly into a UNet with or without CBAM attention. The UNet model here is similar but not identical to the UNet model in our previous work.

*Diagram of all two-stage models (MAE_CBAM, MAE_UNet, Interpolation_CBAM, Interpolation_UNet). Single-stage models (CBAM, UNet) omit the first stage.*

**Important implementation note:** while the above diagram shows a sampled map being fed into the model as input, for training purposes all models actually taken in the *complete* radio map as input and carry out sampling within stage 1 of the model (so the model does not "see" the complete map when making its predictions, but carries out sampling as a pre-processing step within the model). For inference or evaluation, the user can set the hyperparameter *pre_sampled=True* to feed already sampled maps directly to the model, but (1) such maps have to have a batch size of 1, and (2) the sampled maps must have an accompanying mask that specifies sampled locations (if *pre_sampled=False*, the model generates this mask while sampling the complete map). The reason for the batch size of 1, and why training the models on pre-sampled maps isn't an option, is due to limitations in the training set, and could be changed in the future. This will be discussed in more detail in the sections on the MAE sub-model and the dataset below.

## Sub-Model Architectures

The sub-models that make up stages 1 and 2 of the two-stage models are saved in the sub_models folder, and are all preceded with an underscore "_" to distinguish them from similarly named full models. The four sub-models currently implemented are discussed below, with links to relevant papers and github repositories where they have been copied or adapted from other projects.

**MAE** ([paper](https://arxiv.org/abs/2111.06377) | [github](https://github.com/facebookresearch/mae))

The masked autoencoder (MAE) is a masked vision transformer with similar principles to masked language models, e.g. BERT. It takes a complete image, masks out parts of it, and then predicts the masked parts. Our implementation makes a few changes from the linked paper and code above. First, it makes positional embeddings learnable and concatenates these to the input rather than adding them. This is to give the model more flexibility in assigning position embeddings to each pixel, so that it can learn spatial relationships between them. Second, in the decoder that predicts the values of masked pixels, we use cross-attention between the masked and unmasked pixels (rather than concatenating masked and unmasked pixels together and using self-attention between all of them). Intuitively, this means that the masked pixels (i.e. unsampled locations) will not calculate attention with each other, but will only calculate attention with the *sampled locations*. Practically, it means that key and value vectors for cross-attention only have to be calculated for the sampled pixels, while query vectors only have to be calculated for the unsampled pixels. With small numbers of sampled pixels, this reduces GPU memory requirements significantly and allows for faster processing of larger batches. Finally, when sampling the complete map, we take building locations into account so that no samples are drawn from locations occupied by buildings. This allows the MAE to take complete maps as input and sample them itself, or it can be given pre-sampled maps as input (by setting *pre_sampled=True*).

One limitation of the MAE is that it expects all images / maps within a batch to have the same number of sampled pixels. This is possible when sampling is done within the MAE itself, because we can just set the number of samples per batch to a specific number. However, with pre-sampled maps, we can't enforce this unless we bin all maps with the same number of sampled pixels together or set batch size equal to 1. It might be possible to implement the first option with a dataset in the future, or to change the encoder so that it can accept variable-sized inputs within a single batch, but for now we have simply restricted all models to only accept pre-sampled maps when the batch size is 1.

**CBAM** ([paper](https://arxiv.org/abs/1807.06521) | [github](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/README_EN.md#6-cbam-attention-usage))

The Convolutional Block Attention Module (CBAM) is a type of gated attention applied to feature maps in a CNN that squishes some features to 0 and allows others to pass through unchanged (more accurately, it multiplies all features by a factor between 0 and 1). The CBAM sub-model is a UNet with CBAM-style attention. The UNet has three "levels", where each level consists of three convolution + CBAM layers followed by downsampling or upsampling.

CBAM was the first type of attention we considered applying to RME, with the thought that it would learn to emphasize features in areas with more sampled measurements and downplay features in areas with fewer sampled measurements, while also potentially responding to implicitly learned features such as possible transmitter positions. However, since the input is already so sparse (all unsampled locations are filled with 0 by default), an attention mechanism that operates by squishing unimportant values to 0 might not have the same effect here. This gave us the idea of filling in those unsampled locations and creating a "dense" map (i.e. stage 1 of the two-stage model). Our initial idea was to use interpolation to do this (realized in the Interpolation_CBAM model), though we actually started on the more powerful Masked Autoencoder (MAE_CBAM). We also created a CBAM model without any stage 1 to allow the UNet with CBAM to operate directly on the sparse maps as an ablation study.

**UNet**

The UNet sub-model is identical to the CBAM sub-model above, but with the Convolutional Block Attention Module removed. It is used in the MAE_UNet, Interpolation_UNet, and UNet models for ablation studies to see whether the Convolutional Block Attention Module actually improves model performance in any of those cases.

**Interpolation** ([code](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html))

To see whether the Masked Autoencoder (MAE) actually improves model performance, we replace it with nearest neighbor, linear, or cubic interpolation in the Interpolation_CBAM and Interpolation_UNet models. The implementation is taken from SciPy's interpolate.griddata function.

A practical challenge of using SciPy's interpolation function is that it requires putting all input tensors back onto the CPU and calculating the interpolated values for each map sequentially in a for loop. Because of this, models that use interpolation train much more slowly than those that use MAE, despite interpolation not having any learnable weights. If a function allowed multiple maps to be interpolated simultaneously on the GPU, this could speed up training significantly. The issue is raised in a PyTorch feature request [here](https://github.com/pytorch/pytorch/issues/1552), with multiple solutions proposed for different cases, but I am not sure if any would fit our specific case.

## Dataset

One more important consideration before looking at experimental results is the dataset itself. Unlike the dataset used in [previous work](https://github.com/GeoAICenter/radio-map-estimation-pimrc2023), which was generated using code adapted from Deep Completion Autoencoders for Radio Map Estimation ([paper](https://arxiv.org/abs/2005.05964) | [github](https://github.com/fachu000/deep-autoencoders-cartography)), the dataset used here is adapted from the RadioMapSeer dataset introduced in RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks ([paper](https://arxiv.org/abs/1911.09002) | [github](https://github.com/RonLevie/RadioUNet) | [dataset](https://radiomapseer.github.io/)). Both datasets consist of radio maps generated by ray tracing, but with some important differences. The radio maps in the original dataset were 32 x 32 pixels with two transmitters at unknown locations (possibly outside of the 32 x 32 map). The radio maps in the new dataset were originally 256 x 256 pixels, but have been scaled down to either 32 x 32 or 64 x 64 pixels with a single transmitter at a known location (always located somewhere within the map). We generally don't allow the model to know the transmitter location. The new dataset measures pathloss rather than raw power, meaning it measures the difference in decibels between the power sent out at the transmitter and the power received at any given point on the map. For more details on the radio characteristics of the dataset, see Section 3 of the RadioUNet paper; for more details on the formatting and resizing of the dataset, see the following two notebooks: ([1](https://colab.research.google.com/drive/13OAe3_C0efqkn2QdDNzxOt0_K2jIy1BI?usp=sharing)) and ([2](https://colab.research.google.com/drive/16R3HMGuwOnaN3BTr-9L7He-u8j60Ip89?usp=sharing)). You can also find information in the comments of [util.dataset.py](util/dataset.py) in this repository.

The current dataset is saved and passed to the model with the following data: *sampled map*, *complete map*, *building mask*, *filepath*, and *transmitter location*. For compatibility with earlier datasets, the *complete map* is actually passed to the model twice, but both contain identical information. The *sampled map* is a tensor of size 2xHxW, where H and W are the height and width of the map. The first channel contains the radio power (pathloss) scaled between 0 and 1 at sampled locations, with 0 at all unsampled locations. The second channel is a ternary mask, with a value of 1 at sampled locations, -1 at building locations, and 0 at unsampled, non-building locations. As mentioned in the [Model Architectures](#model-architectures) section, the sampled map is only used when *pre_sampled=True* in the forward method; this is because the number of sampled pixels ranges from 1% to 40% (on 32 x 32) or 1% to 20% (on 64 x 64) of the total free space pixels on a given map, and we would have to bin them by the exact number of pixels sampled to pass them in batches to the MAE. Normally the *complete map* is passed to the model, which is then sampled before going through the forward pass (it is also used in un-sampled form as the ground truth). The *complete map* is a 1xHxW tensor, where each location has the radio power (pathloss) scaled between 0 and 1, and the radio power at building locations is 0. The *building mask* is a 1xHxW binary mask with 1 at building locations and 0 elsewhere. The *filepath* and *transmitter location* aren't used in the current models, but *filepath* is a string identifying the pickle file where the data is saved, and *transmitter location* is a size 2 tensor with the x and y coordinates of the transmitter scaled between 0 and 1 as a proportion of the width and height of the map.

Ideally, we will be able to generate new maps tailor made to our problem, but in the meantime all of the experiments below have been carried out with the 64 x 64 RadioMapSeer dataset saved in the Google Drive. If you want to use the full 256 x 256 RadioMapSeer maps, you can consult Thanh Le for his preprocessed dataset, or look at the Colab notebooks in the RadioMapSeer [github](https://github.com/RonLevie/RadioUNet) and [this copied notebook](https://colab.research.google.com/drive/1nSQXgjSuM3-ur-tJxZZhlLRXtRDkP4qd?usp=sharing) with additional notes at the beginning.

## Model Training and Evaluation

When initializing a new model, you will specify the *model_name*, which is used in saving model weights, configurations, and training and validation losses. Other hyperparameters are specific to model type and are specified in the comments to the model's code.

To train a model, you can call the model's *fit* or *fit_wandb* method; the latter will save a record of train and validation losses per epoch in WandB. In the hyperparameters for either method you will need to specify *min_samples* and *max_samples*, which sets the range for the number of samples to be drawn from each map in a batch. For *fit_wandb* you will need to specify *project_name*, which specifies where the loss logs are saved in WandB. For either method, you can choose to specify *run_name*, though if you leave it as *None* the model will create a *run_name* based on the other hyperparameters in the *fit* or *fit_wandb* function, and this is recommended for consistent naming conventions. *dB_max* and *dB_min* are used for converting the scaled pathloss values back to their original values in dB, and the default values are set based on numbers specified in the RadioUNet paper (though looking at the RadioUNet code recently has made me question whether these values are correct). The *free_space_only* hyperparameter specifies whether the predicted radio power at building locations is counted towards the loss or error; in our previous work it was not counted, but in the RadioUNet paper it seemingly is.

For any model with MAE as the first stage, there is a hyperparameter *mae_regularization* that adds a term to the training loss for the MSE between the dense map (output by the MAE in stage 1) and the ground truth map. Otherwise the loss is only calculated between the stage 2 output and the ground truth map, and must be backpropagated through the entire stage 2 model before arriving at the MAE. For any model with Interpolation as the first stage, there is a *method* hyperparameter that specifies whether the interpolation used is nearest neighbor ('nearest'), linear, or cubic.

The *evaluate* method has much the same parameters as *fit* and *fit_wandb*, but you can also specify *pre_sampled=True* to evaluate on pre-sampled maps with a batch size of 1.

## Model Saving and Loading

In the shared Google Drive, models are organized by *model_name* (specified in the *init* method) first and *run_name* (specified in the *fit* or *fit_wandb* method) second. The organizing principle is that models sharing the same initialization parameters will be grouped together under a single folder and then subdivided based on their training parameters. For example, two MAE_CBAM models with identical hyperparameters might both be saved under the *model_name* **Model 1** but separate *run_names* reflecting whether they were trained on 3-41 samples or 42-82 samples, or perhaps with or without *mae_regularization*. 

When the *save_model* method is called, a folder with the *model_name* will be created if one doesn't already exist. It will then save the model's initialization hyperparameters as a json file in that folder. If the *fit* or *fit_wandb* method has already been called, it will then create a *run_name* sub-folder, which is where it will save the training hyperparameters and trained weights. The weights will be saved as a state dictionary named according to the number of epochs the model was trained.

The *save_model* method takes as optional arguments *epochs* to manually specify the number of epochs trained, *optimizer* to save the optimizer's state dictionary if provided, *scheduler* to save the scheduler's state dictionary if provided, and *out_dir* to specify the directory in which the model folder will be created. When *fit* or *fit_wandb* is called, the *save_model* method will automatically be called every *n* epochs as specified by the hyperparameter *save_model_epochs*.

**warning:** If two models with different initialization hyperparameters both share the same *model_name*, the *config* file of the latter will overwrite the *config* file of the former without any warning. Take care that models only share the same name if they share the same hyperparameters.

To load a model, first initialize a new model with the same hyperparameters as the saved model. Then use the *load_state_dict* method to load the weights saved under the appropriate file.

## Experimentation and Results

Thus far we have trained five separate named models, some with multiple different training parameters. Below are some example maps drawn by those models, with the training parameters listed above. In each case, the first image is the ground truth map, the second is the sampled locations, third is the dense map predicted from those sampled measurements, and fourth is the final map convolved with buildings. All models are trained for 50 epochs.

**Model 1: MAE_CBAM**

MAE_CBAM, trained on 3-41 samples, free space only
![Image](images/MAE_CBAM,%203-41%20samples,%20free%20space%20only,%2050%20epochs.png)

MAE_CBAM, trained on 42-82 samples, free space only
![Image](images/MAE_CBAM,%2042-82%20samples,%20free%20space%20only,%2050%20epochs.png)

MAE_CBAM, trained on 83-123 samples, free space only
![Image](images/MAE_CBAM,%2083-123%20samples,%20free%20space%20only,%2050%20epochs.png)

These were the first three models trained to completion (three other MAE_CBAM models were trained for 15 epochs with *free_space_only=False*). On the models trained on 3-41 samples or 42-82 samples per map, stage 1 of the model seems to be learning a spatial model of radio propagation that allows it to predict higher power near the (unknown) transmitter location and lower power farther away from it; this is especially apparent in the 42-82 sample case. However, for some reason at 83-123 samples it seems to lose this capacity, predicting higher signal strength across the entire area, with almost no visible concentration around the transmitter location. Nonetheless, all three predict similar (though not identical) final maps with buildings, with the only visible difference being a slightly blurrier image in the 3-41 sample map.

At Nikita Lokhmachev's suggestion, we added *mae_regularization*, a secondary loss term that calculates the MSE between the dense map output by stage 1 of the model and the ground truth, despite the fact that the dense map prediction doesn't have access to building locations and so presumably would not be able to closely match the actual map with buildings. However, we got the following very surprising results:

MAE_CBAM, trained on 3-41 samples, free space only, mae regularization
![Image](images/MAE_CBAM,%203-41%20samples,%20free%20space%20only,%20mae%20regularization,%2050%20epochs.png)

MAE_CBAM, trained on 42-82 samples, free space only, mae regularization
![Image](images/MAE_CBAM,%2042-82%20samples,%20free%20space%20only,%20mae%20regularization.png)

Despite not having access to building locations, the dense maps output by the MAE were able to model radio propagation (including shadowing from the unseen buildings) with remarkable fidelity. The area of strongest signal is localized much more narrowly around the (still unknown) transmitter position, and the predicted radio power dropoff is almost identical to the final map predictions except for the absence of zero or near-zero power predictions at actual building locations.

This visible difference in the quality of the stage 1 predictions would suggest a similar improvement in stage 2 predictions, but notably we cannot see any such improvement by the naked eye. Looking at validation loss across epochs of all five runs, we observe the following.

![Image](images/MAE_CBAM%20Validation.png)

In fact, the increased fidelity of the stage 1 maps with *mae_regularization* doesn't seem to have any bearing on the final error of the stage 2 maps, which is the only output we care about. The only difference we see is that models trained and evaluated on greater numbers of samples perform better than models trained and evaluated on lower numbers of samples (in each case, the model is evaluated on maps with the same range of samples as it was trained on). Given the significant difference between models trained on 3-41 samples and models trained on 42-82 samples, we decide to train all following models on 42-82 samples as a common comparison point (the performance boost of training on 83-123 samples is much less significant, and these models take longer to train). 

**Model 2: Interpolation_CBAM**

Replacing the stage 1 MAE with simple interpolation (nearest neighbor and linear), we get the following dense and final maps.

Interpolation_CBAM, trained on 42-82 samples, free space only, nearest neighbor interpolation
![Image](images/Interpolation_CBAM,%2042-82%20samples,%20free%20space%20only,%20nearest%20interpolation.png)

Interpolation_CBAM, trained on 42-82 samples, free space only, linear interpolation
![Image](images/Interpolation_CBAM,%2042-82%20samples,%20free%20space%20only,%20linear%20interpolation.png)

The stage 1 dense map with nearest neighbor interpolation, though much blockier and less detailed than the stage 1 MAE dense prediction with *mae_regularizatioin*, is nonetheless still recognizable from the ground truth map, and the final prediction looks reasonably accurate. The stage 1 dense map with linear interpolation, on the other hand, looks almost unrecognizable, but again the final prediction doesn't seem to suffer from this. Comparing their validation losses across epochs confirms this.

![Image](images/Interpolation_CBAM%20Validation.png)

Again, there isn't an apparent advantage gained from either model's stage 1 output on the stage 2 prediction, despite them being so different. And comparing both models' validation loss against the comparable runs of Model 1, we see the same.

![Image](images/MAE_CBAM,%20Interpolation_CBAM%20Validation.png)

**Model 3: MAE_UNet**

Keeping the stage 1 MAE but removing CBAM attention from the UNet in stage 2, we test the difference this makes with the following model. We do not include *mae_regularization* with this model.

MAE_UNet, trained on 42-82 samples, free space only
![Image](images/MAE_UNet,%2042-82%20samples,%20free%20space%20only.png)

Without *mae_regularization*, we're back to the much fuzzier stage 1 dense map prediction, but the main place we're looking for differences is in the final predicted map, since this is output by the UNet now without CBAM attention. Interestingly, the color at building locations is visibly lighter, indicating that the model predicted higher radio powers at buildings than previous models (which correctly predicted them as close to zero); however, because loss is calculated on free space only, this does not affect training or validation score. Looking at validation loss between MAE_UNet and the corresponding MAE_CBAM run, we see the following.

![Image](images/MAE_UNet%20Validation.png)

Again, there is very little difference between the two models and two training runs. It's possible that the MAE_UNet has a slightly jumpier validation loss throughout its 50 epochs of training, but it's difficult to say how significant this is, and the two losses cross over each other multiple times throughout training.

**Model 4 and Model 5: CBAM and UNet**

Given the apparent lack of influence different stage 1 dense maps have on stage 2 predictions, we train Models 4 and 5 without a stage 1 at all, just feeding the sparse input directly into the CBAM and UNet respectively. Model 5 is very similar to how we trained the models in our previous work, but the architecture of the UNet is changed slightly; the architectures of Models 4 and 5 is identical except for the inclusion of CBAM attention in Model 4.

CBAM, trained on 42-82 samples, free space only
![Image](images/CBAM,%2042-82%20samples,%20free%20space%20only.png)

UNet, trained on 42-82 samples, free space only
![Image](images/UNet,%2042-82%20samples,%20free%20space%20only.png)

Because there is no stage 1, the "dense map" just shows the sampled map fed directly into the convolutional models. Again we see that the UNet predicts higher powers at buildings (at least on the largest building) than the CBAM, but again loss is only calculated on free space and so this doesn't affect performance scores. Looking at their validation losses across epochs, we see very similar performance, though the UNet is again perhaps a little jumpier than the CBAM, and both are perhaps slightly jumpier than most two-stage models.

![Image](images/CBAM,%20UNet%20Validation.png)

**Overall**

Looking at all models trained on 42-82 sample maps for 50 epochs, we get the following result.

![Image](images/All%20Models%20Validation.png)

The similarity of validation losses across training would seem to suggest none of the interventions tested had a significant impact on radio map estimation accuracy. The only visible changes are that the UNet, MAE_UNet, and to a lesser degree CBAM do seem to have a slightly jumpier validation loss across different epochs, but again the significance of this is hard to judge from these results. 

It's possible we should still test and compare these models on a fixed validation dataset, e.g. the pre-sampled maps from the validation set with the fewest samples, so we can compare all of them on exactly the same sampled maps. Early tests doing this suggested there might still be an advantage for these models over previous ones at lower sampling rates; however, this might simply be due to the fact that these models were trained exclusively on lower sampling rates (between 1-3% of total numbers of pixels), while previous models were trained on much larger ranges of sampling rates (between 1-20%). Theoretically, it's also reasonable to ask why these interventions (which make sense intuitively, and produce clear differences in stage 1 predictions) don't make a larger difference on final prediction errors, and whether there might be hyperparameter settings that can make more effective use of these interventions.