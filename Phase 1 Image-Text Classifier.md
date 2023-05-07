## Phase 1: Image-Text Classifier

#### Introduction

- Objective: predict the relationship between textual and visual inputs
- Inputs: images and their corresponding text descriptions
- Transform text into text embeddings, and images into image embeddings. Then, define the loss function and train the classifier with the joint embeddings.

#### Code outline

1. Resize all the images to a size of 250x250 pixels.

2. Split the data into training, validation, and test sets.

3. Convert images to tensors and stack them into one tensor.

4. Set up a pre-trained BERT model and tokenizer, to convert text descriptions to tensors.

5. Define the text model, image models, loss function, training, evaluation and testing processes.

6. Execute the training and testing processes. Output the running training and evaluation losses (diagrams), classification accuracy, and the testing accuracy.

#### Text Model

Input sequence and corresponding mask -> Pretrained BERT model -> FC layer (768, 512) -> ReLU activation function -> Drop-out layer (rate = 0.1) -> FC layer (512, 1000) -> LogSoftmax activation function (is this implemented?)

BERT is a deep learning model for NLP tasks, pre-trained on large amounts of unlabelled texts. It consists of multiple Transformer encoder layers and performs two tasks: masked language modelling and next sentence prediction. It is first used to extract features from the input text, and then these features are fed into a task-specific output layer.

Optimization algorithm: AdamW (Adam with Weight Decay regularization) involves adding a penalty term to the loss function that penalizes large weight values. Weight decay is added to the weight update step to ensure that the weight decay does not interfere with the momentum-based updates performed by the Adam optimizer. It also uses a modified learning rate schedule adjusting for the weight decay parameter, to prevent the learning rate from decaying too quickly.

#### Image Model

Several models are tested (choosing criteria?):

- CNN_Model with 2 convolutional layers, 2 fully connected layers, and a max pooling layer

- GoogleNet with a modified final fully connected layer

- ResNet18 with a modified final fully connected layer

- ResNet34 (chosen?) with a modified final fully connected layer

GoogleNet is a pre-trained deep convolutional neural network for image classification and object detection tasks. Inception modules are designed to capture information at different spatial scales. Each module contains parallel convolutional layers of different sizes, and the outputs are concatenated to produce the final output of the module.

#### Loss function

- Loss = Ratio of the average distances between matching text-image pairs to that of non-matching pairs

- Distance can be L2 (nn.PairwiseDistance(p=2), sklearn.metrics.pairwise.euclidean_distances) or cosine (sklearn.metrics.pairwise.cosine_similarity)

#### Training and validation processes

- Training: randomly sample training data of batch size 20 -> calculate the loss -> backpropagate and update the parameters using the AdamW optimizer -> (repeat) return the average loss over all batches and the list of samepairdist (for calculating mean)

- Validation: iterate through validation data by batches of size 20 -> evaluate the performance of the trained model -> (repeat) return the average loss over all batches and the list of samepairdist (for calculating standard deviation) -> choose the best model based on the validation loss

#### Testing process

Testing: iterate through testing data by batches of size 20 -> In test(), topk specifies the number of top results to consider for each input; outtxt, outimg, check, and outindex store the output of the models, the ground truth labels, the predicted labels, and the indices of the predicted labels, respectively. It iterates over each batch of the test data and extracts the text sequences, image tensors, and ground truth labels. It then feeds the text sequences and image tensors to the trained text and image models, respectively, to get the corresponding output embeddings. For each text embedding, the function computes the pairwise distance to all image embeddings using the pdist function, which calculates the Euclidean distance between each pair of input vectors. The function then selects the top k distances and corresponding indices using the torch.topk function. For each text embedding, the function compares the distance between the text embedding and the correct image embedding with the top k distances to determine whether the correct image embedding was correctly identified. If the correct image embedding was not included in the top k predictions, the function sets the value of the corresponding element in the check tensor to True, indicating that the prediction was incorrect. (correct?) Finally, the function returns the image tensors, the check tensor, which contains a boolean value for each input indicating whether the correct image embedding was included in the top k predictions, and the outindex tensor, which contains the indices of the top k predictions for each input. The shapes of the returned tensors are printed to provide information on the dimensions of the tensors.

**Results**

(To be updated)



## Phase 2: Classifier Link with Generated Image

The GAN model will be employed to generate large number of images. Then, the trained classifier will be employed to link up with the closest pair of generated image and inputted text.

 