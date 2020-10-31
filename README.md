# C3_V2
This is the customized version of original C-3-Framework [Link](https://github.com/gjy3035/C-3-Framework).
The usage and details about this framework are shown in the homepage of C-3-Framework.

## Updates

- Density map downscaling is supported to fit the outputs of different networks.
- Trainers are embeded in each corresponding basemodel, modify training components is more flexible.
- The process of adding a new network is simplified. Adding a new network by creating a new network file following the template, maintaining a list of networks' name is no more neccesary.
- Fixed the unexpected key error when loading weights (It happens in the condition of multi GPUs training but single GPU testing).
- CSRNetBN (Batch Normalization) and VGG19 are supported.
- Grid Average Mean Absolute Error (GAME) metric for TRANCOS dataset is supproted.
- TRANCOS dataset is supported.

