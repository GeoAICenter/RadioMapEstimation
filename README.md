# Two-Stage Radio Map Estimation

This repository builds on [previous work](https://github.com/GeoAICenter/radio-map-estimation-pimrc2023) in radio map estimation (RME), predicting radio power across a two-dimensional area. As in previous experiments, we don't have knowledge of transmitter location(s), and the environment contains buildings which block and reflect radio signals. We take as input a small number of radio power measurements and a mask showing the locations of these sampled measurements and the buildings in the environment. The model then outputs a complete radio map showing predicted radio power at all locations in the 2D area (optionally including predictions at building locations).

![Image](images/example_input_output.png)

*Sample Input and Output: Sampled Map and Environment Mask are fed as input to the model, which outputs Complete Radio Map. Input and ground truth maps are adapted from the RadioMapSeer Dataset, discussed below.*

The idea of two-stage radio map estimation is to split the task of RME into two parts: first, taking the sparsely sampled measurements and predicting a dense radio map *without reference to buildings*, as if the signal were propagating through empty space; second, taking this dense radio map prediction and convolving it with a map of building locations ("Building Mask") to produce a more accurate radio map that takes the physical environment into account.

![Image](images/example_input_output_2.png)

*Sample Input and Output of Two-Stage RME: Instead of jumping straight from the sampled map to the complete radio map with buildings, an intermediate map is produced ("Dense Map") that ignores buildings and just extrapolates from sample strength and location.*

The intuition behind this is that radio propagation has unique physical characteristics, in particular a source of propagation (i.e. a transmitter) and a relationship between signal strength and distance from that source, that most image data don't have. A model that can learn these physical characteristics will presumably perform better than one that treats the radio map like a normal image. The first stage of the model is designed to learn these spatial relationships, and the second stage to combine this with environmental information. Our main model, MAE_CBAM, uses a masked vision transformer with learnable position encodings, self-attention, and cross-attention to accomplish the first goal, and a CNN / UNet with CBAM-style gated attention for the second. Other models replace / remove different elements of this main model to carry out ablation studies.

## Model Architectures

The two-stage RME models are composed of two sub-models, one for each stage. As stated above, for the main [MAE_CBAM](models/mae_cbam.py) model, these are a masked vision transformer and a UNet with CBAM gated attention. The masked vision transformer is adapted from the code for Masked Autoencoders Are Scalable Vision Learners ([paper](https://arxiv.org/abs/2111.06377) | [github](https://github.com/facebookresearch/mae/blob/main/models_mae.py)), and the UNet is adapted from our previous code with an added implemention of CBAM: Convolutional Block Attention Module ([paper](https://arxiv.org/abs/1807.06521) | [github](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py)).

For ablation tests, we also have an [MAE_UNet](models/mae_unet.py) model, which removes CBAM attention from the UNet in the second stage of the model; the [Interpolation_CBAM](models/interpolation_cbam.py) and [Interpolation_UNet](models/interpolation_unet.py) models, which replace the masked vision transformer with basic interpolation; and the [CBAM](models/cbam.py) and [UNet](models/unet.py) models which remove the first stage altogether and feed the sparsely sampled map directly into a UNet with or without CBAM attention. The UNet model here is similar but not identical to the UNet model in our previous work.

*Diagram of all two-stage models (MAE_CBAM, MAE_UNet, Interpolation_CBAM, Interpolation_UNet). Single-stage models (CBAM, UNet) omit the first stage.*

**Important implementation note:** while the above diagram shows a sampled map being fed into the model as input, for training purposes all models actually taken in the *complete* radio map as input and carry out sampling within stage 1 of the model (so the model does not "see" the complete map when making its predictions, but carries out sampling as a pre-processing step within the model). For inference or evaluation, the user can set the hyperparameter *pre_sampled=True* to feed already sampled maps directly to the model, but (1) such maps have to have a batch size of 1, and (2) the sampled maps must have an accompanying mask that specifies sampled locations (if *pre_sampled=False*, the model generates this mask while sampling the complete map). The reason for the batch size of 1, and why training the models on pre-sampled maps isn't an option, is due to limitations in the training set, and could be changed in the future. This will be discussed in more detail in the sections on the MAE sub-model and the dataset below.

## Sub-Model Architectures

The sub-models that make up stages 1 and 2 of the two-stage models are saved in the sub_models folder, and are all preceded with an underscore "_" to distinguish them from similarly named full models. The four sub-models currently implemented are discussed below, with links to relevant papers and github repositories where they have been copied or adapted from other projects.

**MAE** ([paper](https://arxiv.org/abs/2111.06377) | [github](https://github.com/facebookresearch/mae/blob/main/models_mae.py))
The masked autoencoder (MAE) is a masked vision transformer.