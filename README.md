#  Neural-Style-Transfer

*COMPANY* : CODETECH IT SOLUTIONS

*NAME* : ANUJ DESHMUKH

*INTERN ID* : CT04DL900

*DOMAIN* : AI

*DURATION* : 4 weeks

*MENTOR* : NEELA SANTOSH

# Description

This code performs **Neural Style Transfer** using **PyTorch**, a technique that applies the artistic style of one image (style image) to the content of another image (content image), producing a new stylized output. The implementation leverages a pre-trained VGG19 convolutional neural network and optimizes an image to minimize content and style loss.

### Device and Image Setup

The script starts by checking for a GPU with torch.cuda.is_available() and assigns it to device for faster computation. It defines an image_loader() function to load and preprocess images. It resizes the image to a square shape (default max size 400), converts it to a tensor, and moves it to the selected device. It also defines imshow() to convert the tensor back to a PIL image and display it using matplotlib.

The content and style images (content.jpg and style.jpg) are loaded, with the style image resized to match the dimensions of the content image to avoid shape mismatch during operations.

### Pretrained Model and Normalization

The VGG19 model, pre-trained on ImageNet, is loaded using torchvision.models.vgg19(pretrained=True).features. This model is used only for feature extraction and not training. Since VGG expects normalized inputs, normalization parameters (mean and standard deviation of ImageNet) are defined and wrapped in a custom Normalization class.

### Loss Modules

Two custom modules define the loss functions:

* **ContentLoss** compares features extracted from a specific VGG layer for the input and content images using Mean Squared Error (MSE).

* **StyleLoss** computes the difference between the **Gram matrices** of feature maps from the input and style images, capturing texture and style information. The gram_matrix() function reshapes and multiplies feature maps to create these matrices.

### Building the Model

get_style_model_and_losses() constructs a new model by copying VGG19's layers one by one. It inserts the normalization module at the beginning and adds ContentLoss and StyleLoss modules after specific convolution layers defined in content_layers and style_layers. As the model is built, the intermediate outputs of the content and style images are captured and used as targets for the respective loss modules.

Once all the required layers are added, the model is truncated just after the last loss module to avoid unnecessary computation.

### Style Transfer Optimization

The run_style_transfer() function performs the optimization. It takes in the content and style images, the VGG model, and an initial image (typically a clone of the content image). It uses the **L-BFGS optimizer**, which is suitable for style transfer due to its stability and efficiency with few parameters.

Within each iteration, a closure function is defined, which clamps the image to keep pixel values within \[0, 1], computes the style and content losses, and performs a backward pass to compute gradients. The total loss is the weighted sum of the style and content losses, controlled by style_weight and content_weight. Typically, style_weight is much higher to ensure stylistic features dominate.

The optimization runs for a set number of steps (num_steps=300 by default), printing the loss values every 50 steps to monitor progress.

### Output and Display

After the optimization is complete, the final image is clamped and returned. The result is displayed using imshow() to visualize the stylized image that blends the content of the original photo with the style of the reference artwork.

### Summary

In essence, this code builds a loss-aware forward model from a pre-trained VGG19 network, then uses gradient-based optimization to update an input image until it resembles the content of one image and the style of another. It’s a hands-on implementation of the neural style transfer algorithm introduced by Gatys et al., showcasing a deep understanding of neural networks, feature extraction, and image processing in PyTorch.



# OUTPUT

![Image](https://github.com/user-attachments/assets/b21d156d-e9f3-4a3c-acdc-cf6fc177a48a)
