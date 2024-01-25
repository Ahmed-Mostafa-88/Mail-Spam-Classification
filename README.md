# Spam Classification Project

## Table of Contents
1. [Introduction](#introduction)
   - [Project Idea](#project-idea)
   - [Data Used](#data-used)
   - [Objectives](#objectives)
2. [Text preprocessing](#text-preprocessing)
   - [Removing HTML Tags](#removing-html-tags)
   - [Tokenization](#tokenization)
   - [Removing Stop Words](#removing-stop-words)
   - [Stemming](#stemming)
3. [Data preprocessing](#data-preprocessing)
   - [Data Splitting](#data-splitting)
   - [Text Vectorization](#text-vectorization)
4. [Model Training](#model-training)
   - [Naive Bayes and Logistic Regression Classification](#naive-bayes-and-logistic-regression-classification)
   - [RNN Models (Simple RNN and LSTM)](#rnn-models-simple-rnn-and-lstm)
5. [Results](#results)
   - [Naïve Bayes](#naïve-bayes)
   - [Logistic Regression](#logistic-regression)
   - [Simple RNN](#simple-rnn)
   - [LSTM](#lstm)
6. [Test](#test)

## 1. Introduction <a name="introduction"></a>
Mail spam classification is really important for keeping our emails safe and organized. It helps to find and stop bad or unwanted emails, like ones that try to trick us into giving personal information or clicking on harmful links. By doing this, it makes sure we only get the emails we really want and keeps our inbox clean and safe. This helps us stay protected while using email and makes it easier for us to see and respond to the emails that matter to us.

### 1.1 Project Idea <a name="project-idea"></a>
Comparing the performance of different models in classifying whether mail is spam or not:
1. Naive Bayes
2. Logistic Regression
3. LSTM
4. Simple RNN

### 1.2 Data Used <a name="data-used"></a>
#### 1.2.1 Source
Kaggle

#### 1.2.2 Data Description
This Data consist of raw mail messages which is suitable for the NLP pre-processing like Tokenizing, Removing Stop words, Stemming and Parsing HTML tags.

### 1.3 Objectives <a name="objectives"></a>
- Explore traditional machine learning approaches for spam classification.
- Evaluate recurrent neural network models for more complex feature extraction.
- Compare the performance of different models and identify the most effective model.

## 2. Text preprocessing <a name="text-preprocessing"></a>

### 2.1 Removing HTML Tags <a name="removing-html-tags"></a>
This step involves eliminating HTML tags present in the text data, by using regular expressions to ensure the content is in a readable text format without any HTML markup.

### 2.2 Tokenization <a name="tokenization"></a>
Tokenization breaks down the text into smaller units, such as words or phrases (tokens). It splits the text into individual meaningful units to facilitate further processing.

### 2.3 Removing Stop Words <a name="removing-stop-words"></a>
Stop words are common words (like "the", "is", "at", etc.) that often don't contribute significantly to the meaning of the text. Removing these words can help in reducing noise and improving the efficiency of analysis or model training.

### 2.4 Stemming <a name="stemming"></a>
Stemming involves reducing words to their root or base form. For instance, words like "running," "runs," and "ran" might all be reduced to the stem "run." This process helps in making similar words to a common form, which can assist in text analysis or classification tasks.

## 3. Data preprocessing <a name="data-preprocessing"></a>

### 3.1 Data Splitting <a name="data-splitting"></a>
The dataset is divided into three separate subsets:
1. Training Set: Contains a significant portion (80%) of the original dataset.
2. Validation Set: Represents a smaller portion (10%) of the original dataset.
3. Testing Set: Also contains a smaller portion (10%) of the original dataset.

### 3.2 Text Vectorization <a name="text-vectorization"></a>
To perform tasks on textual data, it needs to be converted into a numerical format. This process, known as text vectorization, is accomplished using CountVectorizer:
- CountVectorizer:
  - Converts text data into a numerical matrix format.
  - The training set's textual content is transformed into a matrix of token counts.
  - The vocabulary learned from the training data is used to represent the validation and testing data in the same numerical format. This ensures consistency in the way textual data is represented across all sets while preserving the relationship between words and their counts.

## 4. Model Training <a name="model-training"></a>

### 4.1 Naive Bayes and Logistic Regression Classification <a name="naive-bayes-and-logistic-regression-classification"></a>
- Naive Bayes Classifier:
  1. Trains a Naive Bayes classifier using the Multinomial Naive Bayes algorithm.
  2. Uses the count of words to make predictions based on probability. Trained using the training set, then predicts labels for the test set.
- Logistic Regression:
  1. Trains a Logistic Regression classifier.
  2. Makes predictions on the test set and calculates accuracy.

### 4.2 RNN Models (Simple RNN and LSTM) <a name="rnn-models-simple-rnn-and-lstm"></a>
- Tokenizes text data and converts it into numerical sequences for RNN input.
- Pads sequences to ensure a fixed length for RNN models.

#### Simple RNN
- **Embedding Layer:** Represents the initial layer converting words into fixed-size dense vectors. Each word is mapped to a high-dimensional vector.
- **Simple RNN Layer:** Shows the Simple RNN layer with 50 units, describing connections between input and output.
- **Dropout Layer:** Illustrates the dropout layer, indicating the dropout rate of 0.5 (50% dropout).
- **Dense Layer:** Represents the output layer with a single neuron using a sigmoid activation function for binary classification.

![Simple RNN Architecture](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/d5773290-054d-4764-b380-cec6c07b8698)

#### LSTM model
- **Embedding Layer:** Represents the initial layer converting words into fixed-size dense vectors. Each word is mapped to a high-dimensional vector.
- **LSTM Layer:** Shows the LSTM layer with 50 units, describing connections between input, output, and memory cell states.
- **Dropout Layer:** Illustrates the dropout layer, indicating the dropout rate of 0.5 (50% dropout).
- **Dense Layer:** Represents the output layer with a single neuron using a sigmoid activation function for binary classification.

![LSTM Architecture](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/1d42a49c-ddac-490f-b4ae-0470520217f8)

## 5. Results <a name="results"></a>

![Overall Results](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/139db57f-3ec5-439f-a062-d86a4f07fac7)

### 5.1 Naïve Bayes <a name="naïve-bayes"></a>
Naive Bayes Accuracy:
- Accuracy: 97.07%
  - This percentage indicates the overall correctness of the predictions made by the Naive Bayes classifier on the test dataset.
Classification Report:
- Precision: Indicates the ratio of correctly predicted instances of a class to the total instances predicted as that class.
  - For class "0" (Not spam): Precision is 96%, meaning 96% of the predicted "Not spam" instances were actually "Not spam."
  - For class "1" (Spam): Precision is 99%, indicating 99% of the predicted "Spam" instances were actually "Spam."
- Recall (Sensitivity): Denotes the ratio of correctly predicted instances of a class to the actual instances of that class in the dataset.
  - For class "0" (Not spam): Recall is 99%, signifying that 99% of the actual "Not spam" instances were correctly identified.
  - For class "1" (Spam): Recall is 93%, indicating that 93% of the actual "Spam" instances were correctly identified.
- F1-score: The mean of precision and recall, providing a balanced assessment of the classifier's performance for each class.
  - For class "0" (Not spam): F1-score is 98%.
  - For class "1" (Spam): F1-score is 96%.
- Support: Indicates the number of occurrences of each class in the test set.
  - For class "0" (Not spam): There are 376 instances.
  - For class "1" (Spam): There are 204 instances.

![Naïve Bayes Results](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/cd624055-c8f9-49fc-8236-8a3ba006ec5b)

### 5.2 Logistic Regression <a name="logistic-regression"></a>
- The Logistic Regression model achieved a high accuracy of 98.10% on the test dataset, slightly outperforming the Naive Bayes classifier.
- For both "Not spam" and "Spam" classes, the precision, recall, and F1-score values are high, indicating the model's ability to correctly identify both classes with high accuracy and efficiency.
- The classification report suggests that the Logistic Regression model performs exceptionally well in distinguishing between "Not spam" and "Spam" messages, with high precision and recall values for both classes.

![Logistic Regression Results](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/c87855fc-57e7-487c-8dcc-e71735eb9a4e)

### 5.3 Simple RNN <a name="simple-rnn"></a>
- The model shows significant improvement in accuracy and reduction in loss across epochs, indicating the model's learning and improvement over successive iterations.
- The training accuracy consistently increases while the loss decreases with each epoch, demonstrating that the model is learning to predict better on the training data.
- Validation accuracy and loss also show improvements, suggesting that the model is generalizing well to unseen data (validation set) and not overfitting.
- The slight difference between validation and training accuracy/loss in some epochs indicates the model's ability to generalize to new data (validation) while still performing well on the training set.
- The final accuracy on the test set (97.93%) indicates the model's ability to make accurate predictions on previously unseen data.

![Simple RNN Results](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/aff10ed5-b422-495c-8e36-f39897117c4a)

### 5.4 LSTM <a name="lstm"></a>
- The initial training epoch starts with a lower validation accuracy, which gradually increases over epochs.
- There's an increase in training accuracy over epochs, indicating that the model is learning and improving on the training data.
- Validation accuracy generally shows good performance, indicating the model's ability to generalize to unseen data.
- The test set accuracy of 96.72% indicates the model's performance on previously unseen data (test set), demonstrating its ability to make accurate predictions.

![LSTM Results](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/0ef93f71-4d69-4a5b-b8c9-511838433f0b)


## 6. Test <a name="test"></a>
### Example 1:
![Example 1](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/5940ebf9-9e60-4d20-b788-d3233eee7d3f)
- All four models (Naive Bayes, Logistic Regression, LSTM, and Simple RNN) classify this message as "spam."
- The message likely contains phrases often associated with spam content, such as "Congratulations," "free vacation," and "claim your prize," leading the models to categorize it as "spam."

### Example 2:
![Example 2](https://github.com/Ahmed-Mostafa-88/Mail_Spam_Classification/assets/144740078/ef690f69-fae8-45a4-8c94-835da29a8f2f)
1. Naive Bayes: This model predicted the message as ‘not spam’. Naive Bayes classifiers are based on applying Bayes’ theorem with strong independence assumptions between the features. In the context of spam detection, it considers each word in the email independently and estimates the probability of spam given the presence of that word. It seems to have correctly identified that the message “Hi, I am Ahmed.” is not spam.
2. Logistic Regression: This model predicted the message as ‘spam’. Logistic Regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function. The model predicted this simple greeting as spam, it might be due to some bias in the training data where similar phrases were labeled as spam.
3. LSTM (Long Short-Term Memory): This model predicted the message as ‘not spam’. LSTM is a type of recurrent neural network that can learn order dependence in sequence prediction problems. This is important in spam detection as the meaning of a word can often depend on its context and order in the sentence. It seems to have correctly learned that the message “Hi, I am Ahmed.” is a typical non-spam greeting.
4. Simple RNN (Recurrent Neural Network): This model predicted the message as ‘spam’. Simple RNNs also have the ability to use their internal state (memory) to process sequences of inputs, but they can struggle with long-term dependencies due to the vanishing gradient problem. This might have led to the incorrect classification.
