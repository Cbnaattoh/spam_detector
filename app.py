import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('spam_detector.pkl')

# Function to predict if the message is spam or not
def predict_spam(message):
    model = load_model()
    preprocessed_message = preprocess_text(message)
    prediction = model.predict([preprocessed_message])
    # Map prediction to "spam" or "not spam"
    return 'spam' if prediction[0] == 'spam' else 'not spam'

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to train the model and get accuracy
def train_and_get_accuracy(data):
    data['Message'] = data['Message'].apply(preprocess_text)
    X = data['Message']
    y = data['Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load the best model
    model = load_model()
    
    # Predict the category
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function to plot the graph
def plot_data():
    original_data = load_data('email.csv')
    augmented_data = load_data('augmented_email.csv')
    original_counts = original_data['Category'].value_counts()
    augmented_counts = augmented_data['Category'].value_counts()

    counts_df = pd.DataFrame({
        'Original': original_counts,
        'Augmented': augmented_counts
    }).T

    fig, ax = plt.subplots()
    counts_df.plot(kind='bar', ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Count')
    ax.set_title('Number of Spam vs Ham in Original and Augmented Data')
    plt.xticks(rotation=0)
    return fig

# Streamlit web interface
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Spam Detector", "Info"])

    if page == "Home":
        st.title("Home")
        st.write("Welcome to the Spam Detector app!")
        st.write("Use the sidebar to navigate to the spam detection feature.")
        
        # Add a GIF
      

# Path to the GIF file
        gif_path = 'gif/sd.gif'

# Display the GIF
        st.image(gif_path, caption="Spam Detection", use_column_width=True)

        
    elif page == "Spam Detector":
        st.title("Spam Detector")
        
        user_input = st.text_area("Enter a message to check if it's spam or not:", placeholder="Type your message here...")

        if st.button("Predict"):
            if user_input:
                prediction = predict_spam(user_input)
                st.write(f"The message is classified as: {prediction}")
            else:
                st.write("Please enter a message")

    elif page == "Info":
        st.title("Info")
        
     

        st.header("Model Details")
        st.write("""
        **Model**: The spam detection model is trained using a machine learning algorithm. The specific algorithm and hyperparameters are detailed in the model training phase.

        """)

        st.header("Future Improvements")
        st.write("""
        - **Enhance Preprocessing**: Experiment with more advanced text preprocessing techniques.
        - **Model Optimization**: Try different machine learning algorithms or fine-tune hyperparameters for better performance.
        - **Expand Dataset**: Include more diverse datasets to improve the model's generalization capabilities.
        - **User Feedback**: Incorporate user feedback to continuously improve the spam detection accuracy.
        """)
if __name__ == "__main__":
    main()
