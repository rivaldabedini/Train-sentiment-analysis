import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model("sentiment_rnn_model.h5")

# Load the tokenizer
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    messagebox.showerror("File Not Found", "The tokenizer.pkl file was not found.")
    exit()

# Parameters for text preprocessing
max_length = 100

# Function to predict sentiment
def predict_sentiment():
    review = review_text.get("1.0", tk.END).strip()
    if not review:
        messagebox.showwarning("Input Error", "Please enter a review text.")
        return

    # Preprocess the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded_sequence)
    predicted_class = prediction.argmax(axis=1)[0]

    # Map numeric prediction to sentiment label
    reverse_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predicted_label = reverse_mapping[predicted_class]

    # Display the result
    result_label.config(text=f"Predicted Sentiment: {predicted_label}", foreground="blue")

# Create the GUI
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("500x450")
root.resizable(False, False)

# Apply styles
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 11))
style.configure("TFrame", background="#f0f0f0")

# Create and place the widgets
frame = ttk.Frame(root, padding=20)
frame.pack(fill="both", expand=True)

header_label = ttk.Label(frame, text="Sentiment Analysis", font=("Helvetica", 18, "bold"), anchor="center")
header_label.pack(pady=10)

review_label = ttk.Label(frame, text="Enter a review:", anchor="w")
review_label.pack(pady=5, fill="x")

review_text = tk.Text(frame, height=8, width=60, font=("Helvetica", 12), wrap="word", borderwidth=2, relief="groove")
review_text.pack(pady=10)

predict_button = ttk.Button(frame, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack(pady=15)

result_label = ttk.Label(frame, text="", font=("Helvetica", 14), anchor="center", foreground="#333333")
result_label.pack(pady=10)

# Run the GUI event loop
root.mainloop()
