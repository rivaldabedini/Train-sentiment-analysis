from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Step 1: Load the dataset
reviews_data = pd.read_csv("Reviews.csv")
print(reviews_data.columns)

# Step 2: Preprocess the data
# Use 'Text' as input and 'Score' as labels; convert scores into sentiment labels
def map_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

# Create a new sentiment column
reviews_data['Sentiment'] = reviews_data['Score'].apply(map_sentiment)

# Select relevant columns: Text (input) and Sentiment (target)
processed_data = reviews_data[['Text','Score' ,'Sentiment']].dropna()

print(processed_data.head())

# Map sentiment to numerical labels
sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
processed_data['SentimentLabel'] = processed_data['Sentiment'].map(sentiment_mapping)

# Tokenizer parameters
max_words = 10000  # Same as during training
max_length = 100   # Maximum length of a sequence
oov_token = "<OOV>"

# Create and fit the tokenizer
tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
tokenizer.fit_on_texts(reviews_data['Text'])  # Use the 'Text' column of your dataset

# Save the tokenizer for later use
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Convert text to sequences and pad them
sequences = tokenizer.texts_to_sequences(processed_data['Text'])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# save the preprocessed data to a new csv file
processed_file_path = 'preprocessed_reviews.csv'
processed_data[['Text', 'Sentiment', 'SentimentLabel']].to_csv(processed_file_path, index=False)

# Display the processed data
print(processed_data.head())

# Split the data into training and testing sets
X = padded_sequences
y = processed_data['SentimentLabel'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the datasets

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Build the RNN model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model's architecture
print('Model summary',model.summary())
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# Save the trained model
model.save("sentiment_rnn_model.h5")

# Save the preprocessed dataset
processed_data.to_csv("preprocessed_reviews.csv", index=False)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Example new reviews
new_reviews = [
    "This product is amazing! I absolutely love it.",
    "Terrible experience. Will never buy this again.",
    "It's okay, not great but not bad either."
]

# Preprocess the new reviews
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding='post', truncating='post')

# Predict the sentiment
predictions = model.predict(new_padded)
predicted_classes = predictions.argmax(axis=1)

# Map numeric predictions back to sentiment labels
reverse_mapping = {0: "negative", 1: "neutral", 2: "positive"}
predicted_labels = [reverse_mapping[p] for p in predicted_classes]

# Print the results
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {label}\n")
