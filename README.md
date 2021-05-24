## Fast-Neural-Style 🚀

The model uses the method described
in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along
with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf).

**Train:**

There are several arguments to change in `config.yaml`, the important ones are listed below

- `dataset`: path to training dataset, the path should point to a folder containing another folder with all the training
  images. I used COCO 2014 Training images dataset [83K/13GB] [(download)](https://cocodataset.org/#download).
- `style-image`: path to style-image.
- `save-model-dir`: path to folder where trained model will be saved.

**Stylize:**

In `config.yaml`, modify inside `STYLIZE`

- `content-image`: path to content image you want to stylize.
- `model`: saved model to be used for stylizing the image (eg: mosaic.pth)
- `output-image`: path for saving the output image.
- `content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height
  and width of content-image)

**Input image:**

📍 Samarkand, Uzbekistan 

![uzbekistan](images/content-images/uzb.jpg)

**Style image:**

![style](images/style-images/candy.jpg)

**Output image (1 epoch):**

![uzbekistan-candy](images/output-images/uzb_candy_1620970536.jpg)


**Requirements:**

The program is written in Python, and uses [PyTorch](https://pytorch.org/). A GPU is not necessary, but can provide a
significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop
using saved models.

`config.yaml`:

```yaml
TRAIN:
  'num_epochs': 5                                    # Number of training epochs
  'batch_size': 4                                    # Batch size for training
  'dataset': 'Dataset'                               # Path to training dataset
  'style_image': 'images/style-images/mosaic.jpg'    # Path to style images
  'save_model_dir': 'weights'
  'image_size': 256                                  # Train image size, default is 256 X 256
  'style_size':                                      # Style-image size, default is the original size of style image
  'seed': 42
  'content_weight': 1.e+5                            # Weight for content-loss, default is 1e5
  'style_weight': 1.e+10                             # Weight for style-loss, default is 1e10
  'lr': 1.e-3                                        # Learning rate, default is 1e-3
  'log_interval': 500                                # Number of batch intervals to show stats, default is 500

STYLIZE:
  content_image: 'images/content-images/uzb.jpg'     # Path to content image you want to stylize
  content_scale: 1.0                                 # Factor for scaling down the content image, float
  output_path: 'images/output-images/'               # Path for saving the output image
  model: 'weights/candy.pth'                         # If file is .pth - PyTorch

```
