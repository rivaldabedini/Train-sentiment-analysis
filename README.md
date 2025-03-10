Sentiment Analysis Project
==========================

This project implements a sentiment analysis application using a recurrent neural network (RNN). It includes scripts for training and evaluating the model, as well as a graphical user interface (GUI) that lets users analyze text reviews.

Table of Contents
-----------------

*   [Overview](#overview)
    
*   [File Structure](#file-structure)
    
*   [Installation](#installation)
    
*   [Usage](#usage)
    
    *   [Training the Model](#training-the-model)
        
    *   [Evaluating the Model](#evaluating-the-model)
        
    *   [Running the GUI](#running-the-gui)
        
*   [Project Details](#project-details)
    
    *   [Data Preprocessing](#data-preprocessing)
        
    *   [Model Architecture](#model-architecture)
        
    *   [Tokenization and Padding](#tokenization-and-padding)
           

Overview
--------

This project aims to classify text reviews into three sentiment categories: **negative**, **neutral**, and **positive**. It uses an RNN model with an LSTM layer, which is trained on a dataset of reviews. The application provides:

*   A training script (train\_model.py) to preprocess data, build, and train the model.
    
*   An evaluation script (evaluate\_model.py) to test the trained model on saved test data.
    
*   A GUI application (sentiment\_analysis\_gui.py) that allows users to input a review and get an immediate sentiment prediction.
    

File Structure
--------------

*   **train\_model.py** Trains the sentiment analysis model by:
    
    *   Loading and preprocessing review data from Reviews.csv
        
    *   Tokenizing and padding the review texts
        
    *   Building an RNN model with an Embedding layer, LSTM layer, and Dense layers
        
    *   Saving the trained model (sentiment\_rnn\_model.h5), tokenizer (tokenizer.pkl), and preprocessed data
        
*   **evaluate\_model.py** Loads the trained model and test datasets (X\_test.npy and y\_test.npy) to evaluate and print the model's loss and accuracy.
    
*   **sentiment\_analysis\_gui.py** Provides a user interface built with Tkinter that:
    
    *   Loads the trained model and tokenizer
        
    *   Accepts user input for a review
        
    *   Processes the text to predict its sentiment and displays the result
        
*   **requirements.txt** Lists the required Python packages and their versions:
    
    *   numpy==1.21.2
        
    *   pandas==1.3.3
        
    *   scikit-learn==0.24.2
        
    *   tensorflow==2.6.0
        

Installation
------------

1.  **Clone the repository** (or copy the project files into your working directory).
    
2.  **Install Python 3.7+** if you haven’t already.
    
3.  ``` pip install -r requirements.txt ```
    
4.  **Prepare your dataset**:Ensure you have a file named Reviews.csv in the project directory. This file should contain at least the following columns:
    
    *   **Text**: The review content.
        
    *   **Score**: A numeric rating that will be used to derive the sentiment label.
        

Usage
-----

### Training the Model

To train the sentiment analysis model, run:

```python train_model.py ```

This script will:

*   Load the dataset from Reviews.csv.
    
*   Map the numeric review scores to sentiment labels ("negative", "neutral", "positive").
    
*   Tokenize and pad the review texts.
    
*   Split the dataset into training and testing sets.
    
*   Build and train the RNN model.
    
*   Save the trained model as sentiment\_rnn\_model.h5 and the tokenizer as tokenizer.pkl.
    
*   Export the preprocessed reviews to preprocessed\_reviews.csv.
    

### Evaluating the Model

Once the model is trained, you can evaluate its performance with:

``` python evaluate_model.py ```

This script loads the saved model and test data (stored as X\_test.npy and y\_test.npy), evaluates the model, and prints out the loss and accuracy.

### Running the GUI

To start the sentiment analysis GUI, run:

``` python sentiment_analysis_gui.py ```

The GUI window will open, allowing you to:

*   Input a review text.
    
*   Click the "Predict Sentiment" button to analyze the review.
    
*   View the predicted sentiment (Negative, Neutral, or Positive) displayed in the interface.
    

Project Details
---------------

### Data Preprocessing

*   **Dataset Loading**:The train\_model.py script loads review data from Reviews.csv and uses the "Text" column for input and "Score" for labels.
    
*   **Sentiment Mapping**: Scores are converted to sentiment labels:
    
    *   Scores ≤ 2 are labeled as **negative**.
        
    *   A score of 3 is labeled as **neutral**.
        
    *   Scores > 3 are labeled as **positive**.
        
*   **Saving Processed Data**: The preprocessed data is saved to preprocessed\_reviews.csv for future reference.
    

### Model Architecture

*   **Embedding Layer**: Transforms words into dense vectors of fixed size.
    
*   **LSTM Layer**: Captures the sequential dependencies in the text data.
    
*   **Dense Layers**: Provide the final classification into three sentiment categories using a softmax activation function.
    

### Tokenization and Padding

*   **Tokenizer**: The text is converted into sequences using TensorFlow's Tokenizer with an out-of-vocabulary token (). The tokenizer is saved as tokenizer.pkl for use in both training and the GUI.
    
*   **Padding**: Sequences are padded to a maximum length of 100 to ensure consistent input size for the model.
