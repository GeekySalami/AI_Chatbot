import imaplib
import email
import time
import smtplib
from email.mime.text import MIMEText
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

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

# Function to load Training Data from a JSON file
def load_qa_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {item['question']: item['answer'] for item in data}

# Function to send email
def send_email(to_email, subject, body):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "YOUR_EMAIL_ID" 
    password =  "YOUR_APP_PASSWORD" 

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
        logging.info(f"Response sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")
        return False

# Function to process incoming email
def process_email(email_message, qa_data):
    subject = email_message['subject']
    if email_message.is_multipart():
        for part in email_message.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                break
    else:
        body = email_message.get_payload(decode=True).decode()

    question = body.strip()

    best_match = match_question(question, qa_data.keys())
    if best_match:
        response = qa_data[best_match]
    else:
        response = "I'm sorry, I don't understand that question. Can you please rephrase or ask something related to the internship program?"

    sender_email = email.utils.parseaddr(email_message['from'])[1]
    send_email(sender_email, f"Re: {subject}", response)

# Main function to run the email server
def run_email_server(imap_server, email_address, password, qa_data):
    while True:
        try:
            mail = imaplib.IMAP4_SSL(imap_server)
            mail.login(email_address, password)
            mail.select('inbox')

            _, message_numbers = mail.search(None, 'UNSEEN')
            
            for num in message_numbers[0].split():
                _, msg = mail.fetch(num, '(RFC822)')
                email_body = msg[0][1]
                email_message = email.message_from_bytes(email_body)
                process_email(email_message, qa_data)

            mail.logout()
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

        time.sleep(20)

if __name__ == "__main__":
    #Load Data
    qa_data = load_qa_data('output_file_restructured.json')

    IMAP_SERVER = "imap.gmail.com"
    EMAIL_ADDRESS = "YOUR_EMAIL_ID"  
    PASSWORD = "YOUR_APP_PASSWORD"  

    logging.info("Starting email chatbot server...")
    run_email_server(IMAP_SERVER, EMAIL_ADDRESS, PASSWORD, qa_data)