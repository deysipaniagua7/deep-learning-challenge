**This is the repository that holds my Module 21 deep-learning-challenge**  
**Module Overview:** For this assignment, I had to create a tool to help the nonprofit foundation, Alphabet Soup, select the applicants for funding with the best chance of success in their ventures. I had to use the features provided in our dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. The assignment was broken down into 5 steps:

**Step 1: Preprocess the Data**  
- In this step, I identified the target and feature variables in the dataset and drop the “EIN” and “Name” columns.  
- Next, I determined the number of unique values in each column.  
- Then, for columns with 10 unique values, I determined the number of data points for each unique value.  
- I created an “Other” value to contain rare categorical variables.  
- Using the preprocessed data, I created feature and target arrays and then, split the data into training and testing datasets.  
- Finally, the data was scaled using StandardScaler fitted to the training data.

**Step 2: Compile, Train, and Evaluate the Model**  
- In this step, I created a neural network model with a defined number of input features and nodes for each layer.  
- I added 2 hidden layers and an output layer as follows:  
  1) “Dense” layer, units = 80, input_dim = 43, activation = ““relu””;  
  2) “Dense_1, units = 30, activation = 30; and  
  3) “Dense_2”, units = 1, activation = ““sigmoid””.  
- The model was then compiled and trained with epochs = 100.  
- Next, the model was evaluated using the test data for an accuracy of 0.7287.  
- The model was then saved to an HDFS file.

**Step 3: Optimize the Model:** In this step, I repeated the preprocessing steps in a new Jupiter notebook. I also created a new neural network model and implemented 5 model optimization methods as follows:

**Model 1:**  
- I checked for missing data, dropped rows where STATUS=1 and SPECIAL_CONSIDERATIONS=”N”, then dropped these columns, and added a 3rd hidden layer.  
- Individual layer information is below:  
  - Dense: 128, “relu” activation  
  - Dense_1: 64, “relu” activation  
  - Dense_2: 32, “relu” activation  
  - Dense_3: 1, “sigmoid” activation  
- I set the epochs to 100, accuracy was 0.7292.

**Model 2:**  
- I kept everything the same as Model 1 except, changed Dense_1 hidden layer activation from “relu” to “tanh” and increased epochs to 150, accuracy was 0.7291.

**Model 3:**  
- I Added a fourth layer.  
- Individual layer information is below:  
  - Dense: 128, “relu” activation  
  - Dense_1: 64, “tanh” activation  
  - Dense_2: 32, “relu” activation  
  - Dense_3: 16, “relu” activation  
  - Dense_3: 1, “sigmoid” activation  
- I set the epochs to 200, accuracy was 0.7289.

**Model 4:**  
- I kept everything the same as Model 3, except I added 0.2 dropout to reduce overfitting after the first hidden layer and after the second.  
- I then compiled the model and added a lower learning rate of 0.0005.  
- Finally, I added a batch size of 32 before training the model.  
- Epochs remained at 200 and accuracy was 0.7291

**Model 5:**  
- I kept everything the same as Model 4, except this time, I changed the fourth hidden layer’s activation from “relu” to “tanh” and the epochs to 250.  
- The accuracy was 0.7297.

**Step 4: Write a Report on the Neural Network Model:**  
- I wrote a report containing an overview of the analysis, results, and a summary.

**Step 5: Copy Files Into Your Repository:**  
- Finally, I downloaded the Colab notebooks, moved them into my Deep Learning Challenge directory, and pushed the changes to GitHbub.

**Repo Breakdown:** The main “deep-learning-challenge” repo contains:  
- 1) “Starter_Code” subfolder which holds my “Starter_Code.ipynb” code  
- 2) “AlphabetSoupCharity.h5” file which holds the results from step 2  
- 3) “AlphabetSoupCharity_Optimization.ipynb” optimization code  
- 4) “AlphabetSoupCharity_Optimization.h5” files which holds the results from step 3  
- 5) “Deep Learning Challenge Report”

**Please note, I used in-class activities/notes and the following resources to complete my assignment:**  
- https://playground.tensorflow.org/#activation=“sigmoid”&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.3&regularizationRate=0&noise=0&networkShape=4,4,4,2&seed=0.93298&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false  
- https://stackoverflow.com/questions/57076930/how-can-i-prevent-overfitting-in-this-model  
- https://www.tensorflow.org/tutorials/keras/overfit_and_underfit  
- https://ai.stackexchange.com/questions/46143/learning-rate-greater-than-0-00005-significantly-hinders-model-performance-and  
- https://stackoverflow.com/questions/71927467/is-there-a-good-rule-out-there-to-choose-an-appropriate-batch-size?utm_source=chatgpt.com  
- https://stackoverflow.com/questions/70289058/tensorflow-manually-selecting-the-batch-when-training-deep-neural-networks
