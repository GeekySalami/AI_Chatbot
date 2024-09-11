import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json



lemmatizer = WordNetLemmatizer()

# Function to preprocess the text 
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  
    return ' '.join(tokens) 

# Function to find the best matching question based on cosine similarity
def match_question(user_input, questions):
    processed_user_input = preprocess_text(user_input)
    
    questions_list = list(questions)  
    processed_questions = [preprocess_text(q) for q in questions_list]
    
   
    vectorizer = TfidfVectorizer().fit_transform([processed_user_input] + processed_questions)
    similarity_scores = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    
    best_match_index = similarity_scores.argmax()  
    if similarity_scores[best_match_index] > 0.2: 
        return questions_list[best_match_index]
    else:
        return None

# Function to load question-answer pairs from a JSON file
def load_qa_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {item['question']: item['answer'] for item in data}

# Function to check if the user wants to exit
def is_exit(user_input):
    exit_phrases = ['exit', 'bye', 'goodbye', 'see you', 'farewell']
    return any(phrase in user_input.lower() for phrase in exit_phrases)

# Main chatbot function
def chatbot(qa_data):
    print("Welcome to the Python PYQ chatbot! Type 'exit' to stop the chat.")
    
    while True:
        user_input = input("You: ")
        
        if is_exit(user_input):
            best_match = match_question(user_input, qa_data.keys())
            if best_match:
                print(f"Chatbot: {qa_data[best_match]}")
            else:
                print("Chatbot: Goodbye! Feel free to come back if you have more questions.")
            break
        
        # Find the best matching question
        best_match = match_question(user_input, qa_data.keys())
        
        if best_match:
            print(f"Chatbot: {qa_data[best_match]}")
        else:
            print("Chatbot: I'm sorry, I don't understand that question. Can you please rephrase or ask something related to the internship program?")

# Run the chatbot with JSON data
if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('wordnet')
    
    # Load question-answer pairs from the JSON file
    qa_data = load_qa_data('output_file_restructured.json')
    chatbot(qa_data)