# Product Similarity Recommendation

This project aims to recommend products based on their similarity by using both textual and visual features. The visual features are extracted using ResNet-50, while the textual features are extracted using TinyBERT-6L. Preprocessing of images is performed by removing backgrounds using a UNet model trained on fashion pictures (clothes, bags, and shoes).

## Project Overview

In this project, we recommend products based on their similarity. We use cosine similarity to find the similarity between products.

For each product, we extract the visual features and the textual features and then concatenate the two vectors together to build a single vector. This vector is then stored in the vector database, where the cosine similarity is calculated between this vector and the other vectors using a specific technique.

Before extracting the visual data, we preprocess the picture using the "Remove background" model. This model removes any distortion from the image and replaces it with white pixels, ensuring the aspect ratio is conserved even after applying square resizing.

## Project Steps

1. **Import the product data**: Contains the product details and the image URL.
2. **Import the required libraries and models**.
3. **Data Preprocessing**: Removing the background from the images.
4. **Feature Extraction**:
   - Visual features using ResNet-50.
   - Text features using TinyBERT-6L.
5. **Dimensionality Reduction**: Transform the visual and text features into 512-dimensional vectors using PCA.
6. **Normalization**: Normalize the visual and text feature vectors and concatenate the two vectors.
7. **Vector Storage**: Store the vectors in the vector database.
8. **Visualization**: Show the top 7 similar products for a given product ID.

## Repository Structure

The project is divided into the following notebooks:

1. **UNet Model Training**: Training the UNet model for background removal.
2. **Background Remover Function**: Implementing the function to remove backgrounds using the trained UNet model.
3. **Annotation Conversion**: Converting annotations to a suitable format for training and evaluation.
4. **Similarity Recommendation**: The main notebook for generating product recommendations based on similarity.

## UNet Model

UNet is a type of convolutional neural network (CNN) designed primarily for biomedical image segmentation but has found applications in various image segmentation tasks, including fashion item segmentation. Here's how it works:

### Architecture

- **Encoder**: The encoder is a series of convolutional layers followed by max-pooling layers that capture the contextual information and reduce the spatial dimensions of the input image (downsampling).
- **Bottleneck**: This is the layer between the encoder and decoder that processes the most abstract features.
- **Decoder**: The decoder consists of upsampling layers that increase the spatial dimensions of the features, followed by convolutional layers that reconstruct the segmented image (upsampling).
- **Skip Connections**: These connections between corresponding layers in the encoder and decoder allow the model to retain spatial information that may be lost during downsampling, leading to more precise segmentation.

### Segmentation Process

- **Input**: The input image (containing items like clothes, bags, and shoes) is fed into the UNet model.
- **Output**: The model outputs a segmented image where each pixel is classified as belonging to either the foreground (items) or the background.

## Background Removal Function

This function removes the background (anything except for clothes, bags, and shoes) of an image using the trained UNet model. 

### Parameters

- **prob**: Sets the probability threshold for the mask. Default is "argmax" which assigns the pixel to the greater probability class (0: background, 1: foreground). Custom values can also be set.
- **enlarge**: Enlarges the result picture by a certain factor to ensure the object is fully within the picture boundaries (256 x 256). Default is `True`.
- **kernel**: Sets the kernel size of the morphological operation used to remove noise from the mask. Default is `3`.

Applying the mask to the image will remove the background and keep the object in the image.

## Usage

To use this project, follow these steps:

1. **Clone the repository**.
2. **Install the required dependencies**.
3. **Run the notebooks** in the specified order:
   1. `unet_model_training.ipynb`
   2. `background_remover_function.ipynb`
   3. `annotation_conversion.ipynb`
   4. `similarity_recommendation.ipynb`
4. **Visualize the results** to see the top 7 similar products for a given product ID.

## Conclusion

This project provides a comprehensive approach to recommending products based on both textual and visual similarities. By combining powerful feature extraction techniques and efficient preprocessing, it delivers accurate and relevant product recommendations.
