# Modelling Solar Images with DDPM
Github repository for the paper "Modelling Solar Images from SDO/AIA with Denoising Diffusion Probabilistic Models"

## Abstract

Heliophysics lacks enough X-flare data to train a supervised algorithm to forecast flares. The data set is skewed due to the rare physical events, resulting 
in a unbalanced dataset. Generative deep learning models may be used to 
produce high-quality images in different solar activity levels and overcome 
this issue. In this work we train a denoising diffusion probabilistic model. Data is forward 
diffused with Gaussian noise and then the model steadily reverses diffusion 
and recovers input data. AIA instrument data are used and, in particular, the 
17.1 nm band which shows coronal loops, filaments, flares, and active regions. 
Additional channels may be used. The GOES spacecraft X-ray measurements 
were utilised to classify each image using the solar flare scale (A, B, C, M, X), 
together with the Classifier Free Guidance to guide the generation. We labeled the GOES solar flare scale: A, B, C, M and X. We used the HEK dataset to select the flaring events.
Cluster metrics, FID and the F1-score evaluated our generator. Cluster metrics can compare 
the generated image distribution to the real image distribution. To compute 
the FID we used CLIP and IV3 as feature extractors. We show state of the art results on the image generation for Sun images and implement an application to use these generated images on training a supervised classifier
with the addition of the synthetic samples to demonstrate their effectiveness in managing the unbalancing of the dataset.
As future work, we want to better comprehend the generation capabilities of the Denoising Diffusion Probabilistic models and apply them to other deep learning and physical tasks (e.g., AIA to HMI translation, solar flare prediction) being able to investigate these
phenomena more extensively with more data.

## Architecture

![Unet Architecture](https://github.com/Piogeon/Modelling_Solar_Image_with_DDPM/blob/main/images/unet.png)

![Forward and Backward process](https://github.com/Piogeon/Modelling_Solar_Image_with_DDPM/blob/main/images/diffusion.png)
