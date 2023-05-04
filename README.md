# MSDM5005 Project

## Our Project Target:
- An application of deep leardning models: Text-to-Image Synthesis
- Input a text, a description of a image you want. For example, "a bike with a red wheel".
- Our model system will generate a set of images with the content based on the description.

## Datasets
- 102 Category Flower Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
- Caltech-UCSD Bird(CUB): https://drive.google.com/u/0/uc?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE&export=download

## Architecture

=======

- text embeddings
- image embeddings
- text-image classifier (design our loss function)

<img width="291" alt="Screenshot 2023-05-04 at 8 15 23 PM" src="https://user-images.githubusercontent.com/132570829/236211115-a3b3a3c4-acab-46e4-b21a-a25f2020b8a6.png">

=======

- images generation
- GAN / Diffussion model

<img width="457" alt="Screenshot 2023-05-04 at 8 56 01 PM" src="https://user-images.githubusercontent.com/132570829/236211231-f7237ed4-e934-4309-99dd-bdcfd619c3e6.png">

=======

- model integration
- web page for demo

=======

===problems===

- use torch to manage datasets
- call me if need model training, we have 3090 / 4070
- use pytorch lightning for consice training code. (optional)
- limited to our hardware condition, we will try to reduce the model complexity


=======

## Folder structure

- datasets: put the train / validation dataset here
- demo: code for model integration / demostration
- image_generator: GAN model implementation / training code
- text_image_classifier: model implementation / training code
- trained_models: the final trained models (.pt files)

## References

- https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/
- https://repositorio.pucrs.br/dspace/bitstream/10923/18355/2/Efficient_Neural_Architecture_for_TexttoImage_Synthesis.pdf
