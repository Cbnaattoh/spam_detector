import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random
from nltk.corpus import wordnet
import nltk

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: 
            break
    sentence = ' '.join(new_words)
    return sentence

def add_word(words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1 and counter < 10:
        random_word = words[random.randint(0, len(words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
    if synonyms:
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(words)-1)
        words.insert(random_idx, random_synonym)

def random_insertion(sentence, n):
    words = sentence.split()
    for _ in range(n):
        add_word(words)
    return ' '.join(words)

def swap_word(words):
    random_idx_1 = random.randint(0, len(words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1 and counter < 3:
        random_idx_2 = random.randint(0, len(words)-1)
        counter += 1
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    return words

def random_swap(sentence, n):
    words = sentence.split()
    for _ in range(n):
        words = swap_word(words)
    return ' '.join(words)

def random_deletion(sentence, p):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        return words[random.randint(0, len(words)-1)]
    return ' '.join(new_words)

def augment_text(sentence):
    augmentations = [synonym_replacement, random_insertion, random_swap, random_deletion]
    augmented_sentences = [aug(sentence, 1) for aug in augmentations]
    return augmented_sentences

def augment_spam_data(spam_messages, num_augmented=5):
    augmented_data = []
    for message in spam_messages:
        augmented_data.extend(augment_text(message))
    return augmented_data

# Load your data
data = pd.read_csv('email.csv')
spam_messages = data[data['Category'] == 'spam']['Message'].tolist()

# Generate augmented data
augmented_spam_messages = augment_spam_data(spam_messages)

# Append the augmented data to your original data
augmented_data = pd.DataFrame({'Message': augmented_spam_messages, 'Category': ['spam'] * len(augmented_spam_messages)})
augmented_data = pd.concat([data, augmented_data], ignore_index=True)

# Save the augmented data
augmented_data.to_csv('augmented_email.csv', index=False)

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

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def train_model(data):
    # Assuming the dataset has 'Message' and 'Category' columns
    data['Message'] = data['Message'].apply(preprocess_text)
    X = data['Message']
    y = data['Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create a pipeline that combines TfidfVectorizer and MultinomialNB
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    
    # Hyperparameter tuning
    parameters = {
        'vectorizer__max_df': [0.75, 1.0],
        'vectorizer__min_df': [1, 2],
        'classifier__alpha': [0.01, 0.1, 1.0]
    }
    
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'spam_detector.pkl')
    
    # Print the accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')

def predict_spam(message):
    # Load the trained model
    model = joblib.load('spam_detector.pkl')
    
    # Preprocess the message
    preprocessed_message = preprocess_text(message)
    
    # Predict the category
    prediction = model.predict([preprocessed_message])
    return prediction[0]

if __name__ == "__main__":
    file_path = 'augmented_email.csv'  # Update with the correct path to your dataset
    data = load_data(file_path)
    train_model(data)
    
    # Example usage of predicting spam
    user_input = input("Enter a message to check if it's spam: ")
    prediction = predict_spam(user_input)
    print(f'The message is classified as: {prediction}')
