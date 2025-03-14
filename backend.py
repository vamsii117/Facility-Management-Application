import openai
import streamlit as st
from PIL import Image
import io
# from fuzzywuzzy import fuzz, process
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.orm import Session
from datetime import datetime
import speech_recognition as sr
import uuid
# from dotenv import load_dotenv
import sqlite3
import plotly.express as px
import os
# import cv2
import re 
import base64
import numpy as np
import pandas as pd
from numpy.linalg import norm
from googletrans import Translator 

import streamlit as st
from twilio.rest import Client

# Fetch secrets from Streamlit Cloud
openai.api_key = st.secrets["OPENAI_API_KEY"]
TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_NUMBER = st.secrets["TWILIO_WHATSAPP_NUMBER"]
WORKER_WHATSAPP_NUMBER = st.secrets["WORKER_WHATSAPP_NUMBER"]

'''
# environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

#for the whatsapp
from twilio.rest import Client

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
WORKER_WHATSAPP_NUMBER = os.getenv("WORKER_WHATSAPP_NUMBER")  # ✅ Single predefined number
'''

def send_whatsapp_message(worker_name, issue_type, description):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message_body = f"⚡ New Issue Assigned ⚡\n\nWorker: {worker_name}\nIssue Type: {issue_type}\nDescription: {description}\nPlease check the system for more details."

    try:
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=WORKER_WHATSAPP_NUMBER  # ✅ Ensure this number is verified in Twilio sandbox
        )
        return message.sid  # Returns the message ID if successful
    except Exception as e:
        print(f"Twilio Error: {e}")
        return None

###################################################################################################
# for the graph
def fetch_issues():
    conn = sqlite3.connect("tickets.db")
    df = pd.read_sql_query("SELECT *, CASE WHEN resolved = 0 THEN 'pending' ELSE 'completed' END AS status FROM issues", conn)
    conn.close()
    return df

def fetch_issue_data():
    conn = sqlite3.connect("tickets.db")
    query = """
    SELECT 
        DATE(created_at) AS date,
        COUNT(*) AS raised,
        SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) AS completed
    FROM issues
    GROUP BY DATE(created_at)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["date", "raised", "completed", "pending"])  # Return empty DataFrame

    df["pending"] = df["raised"] - df["completed"]
    return df


###################################################################################################
# SQLite database setup
engine = create_engine('sqlite:///tickets.db', echo=True)
Base = declarative_base()

# SQLAlchemy models
class Ticket(Base):
    __tablename__ = 'tickets'
    ticket_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default='open')
    created_at = Column(DateTime, default=datetime.now)
    user_identifier = Column(String)
    issues = relationship("Issue", back_populates="ticket")

class Issue(Base):
    __tablename__ = 'issues'
    issue_id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String, ForeignKey('tickets.ticket_id'))
    type = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    assigned_worker_id = Column(Integer, ForeignKey('workers.worker_id'), nullable=True)
    resolved = Column(Boolean, default=False)  # New Column to track individual issue resolution

    ticket = relationship("Ticket", back_populates="issues")
    assigned_worker = relationship("Worker", backref="issues")

class Worker(Base):
    __tablename__ = 'workers'
    worker_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    specialization = Column(String)
    availability = Column(Boolean, default=True)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Initializing workers.
def initialize_workers():
    existing_workers = session.query(Worker).count()
    if existing_workers == 0:
        workers = [
            Worker(name="John Doe", specialization="electrical", availability=True),
            Worker(name="Jane Smith", specialization="electrical", availability=True),
            Worker(name="Mike Johnson", specialization="plumbing", availability=True),
            Worker(name="Emily Davis", specialization="plumbing", availability=True),
            Worker(name="Manager", specialization="other", availability=True)
        ]
        session.add_all(workers)
        session.commit()

initialize_workers()
###################################################################################################
import streamlit as st
import openai
import base64
from PIL import Image
import io
import numpy as np

# List of known AC companies
known_ac_companies = ["Daikin", "Mitsubishi", "Samsung", "LG", "Whirlpool", "Voltas", "Hitachi", "Panasonic"]

def encode_image(image_bytes):
    """Encodes image bytes to a base64 string for OpenAI API."""
    return base64.b64encode(image_bytes).decode("utf-8")

def extract_text_from_image(image_bytes):
    """Uses OpenAI GPT-4o to extract text from an image."""
    try:
        base64_image = encode_image(image_bytes)
        client = openai.Client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract any readable brand name or text from the given image."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image and return only the text:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300
        )

        # Validate response structure
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            if content:
                return content
        
        print("No content found in response:", response)
        return "No text extracted from the image."

    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return "Error occurred while processing the image."


def identify_ac_company(extracted_text):
    """Matches extracted text with known AC company names."""
    for company in known_ac_companies:
        if company.lower() in extracted_text.lower():
            return company
    return None  # No matching company found

def get_image_embedding(image):
    """Generate OpenAI image embedding for similarity comparison."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=image_bytes
    )
    
    return np.array(response['data'][0]['embedding'])

def identify_company_by_image(uploaded_image, sample_images):
    """Identify AC company using OpenAI embeddings if text extraction fails."""
    uploaded_embedding = get_image_embedding(uploaded_image)

    best_match = None
    highest_similarity = 0.0

    for company, sample_image in sample_images.items():
        sample_embedding = get_image_embedding(sample_image)
        similarity = np.dot(uploaded_embedding, sample_embedding) / (np.linalg.norm(uploaded_embedding) * np.linalg.norm(sample_embedding))

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = company

    return best_match if highest_similarity > 0.7 else "Unknown"

def get_common_issues(company):
    """Generates common issues dynamically for the detected AC brand using OpenAI."""
    prompt = f"List 5 common issues for {company} air conditioners with a short description. Format: 'Issue Name - Description' (e.g., 'Not Cooling - AC is running but not cooling properly')."

    client = openai.Client()  # Initialize the OpenAI client (v1.0+ syntax)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    issues_text = response.choices[0].message.content
    issues = [issue.strip() for issue in issues_text.split("\n") if issue]

    return issues if issues else ["No common issues found."]


###################################################################################################
# get worker issues
def get_worker_issues(worker_id):
    pending_issues = session.query(Issue).filter_by(assigned_worker_id=worker_id, resolved=False).all()
    completed_issues = session.query(Issue).filter_by(assigned_worker_id=worker_id, resolved=True).all()
    return pending_issues, completed_issues

# Assign Worker to an Issue
def assign_worker(issue_type, user_identifier,description1):
    
    # Check if the user has any unresolved issues of the same type
    existing_issue = (
        session.query(Issue)
        .join(Ticket)
        .filter(
            Ticket.user_identifier == user_identifier,
            Issue.type == issue_type,
            Issue.resolved == False
        )
        .order_by(Issue.created_at)
        .first()
    )
    
    # If an unresolved issue exists, assign the same worker
    if existing_issue and existing_issue.assigned_worker_id:
        assigned_worker_id = existing_issue.assigned_worker_id
        assigned_worker_name = session.query(Worker).filter_by(worker_id=assigned_worker_id).first().name
        return assigned_worker_id, assigned_worker_name

    # Find the next available worker with the required specialization
    available_worker = session.query(Worker).filter(
        Worker.specialization == issue_type,
        Worker.availability == True
    ).order_by(Worker.worker_id).first()
    
    if available_worker:
        assigned_worker_id = available_worker.worker_id
        assigned_worker_name = available_worker.name
        
        # Mark the worker as unavailable
        available_worker.availability = False
        session.commit()
        send_whatsapp_message(assigned_worker_name, issue_type, description1) # for whats
    else:
        # No worker is available, the issue will be unassigned
        assigned_worker_id = None
        assigned_worker_name = "No workers available. Issue will be assigned once one is free."
    
    return assigned_worker_id, assigned_worker_name

# Resolving issue for worker dashboard
def resolve_worker_issue(issue_id, worker_id):
    # Fetch the issue to be resolved
    issue = session.query(Issue).filter_by(issue_id=issue_id, assigned_worker_id=worker_id, resolved=False).first()
    if not issue:
        return "Issue not found or already resolved."

    # Mark issue as resolved
    issue.resolved = True
    session.commit()

    # Fetch the worker details
    worker = session.query(Worker).filter_by(worker_id=worker_id).first()
    if not worker:
        return "Worker not found."

    # Check if the user has another unresolved issue of the same type
    existing_issue = (
        session.query(Issue)
        .filter(
            Issue.type == issue.type,
            Issue.resolved == False,
            Issue.assigned_worker_id == worker_id
        )
        .first()
    )

    if existing_issue:
        # Ensure same worker is assigned
        existing_issue.assigned_worker_id = worker_id
    else:
        # Make worker available if no similar unresolved issues exist
        worker.availability = True
        session.commit()

        # Auto-assign to the oldest unassigned issue
        pending_issue = (
            session.query(Issue)
            .filter(
                Issue.type == worker.specialization,
                Issue.assigned_worker_id == None
            )
            .order_by(Issue.created_at)
            .first()
        )

        # Assign the freed worker to the pending issue
        if pending_issue:
            pending_issue.assigned_worker_id = worker.worker_id
            worker.availability = False
            session.commit()
            return f"Issue {issue_id} resolved! Worker {worker.name} reassigned to Issue {pending_issue.issue_id}."
        
        return f"Issue {issue_id} resolved. Worker {worker_id} is now available."

    session.commit()
    return f"Issue {issue_id} resolved. Additional issue merged under the same worker {worker_id}."

import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from googletrans import Translator

# Language Mapping
LANGUAGE_MAP = {
    "English": "en-US",
    "Chinese": "zh-CN",
    "Malay": "ms-MY",
    "Tamil": "ta-IN"
}

TRANSLATE_MAP = {
    "Chinese": "zh-cn",
    "Malay": "ms",
    "Tamil": "ta"
}
from streamlit_js_eval import streamlit_js_eval
def speech_to_text(language):
    """Uses browser-based speech recognition and returns text output."""
    language_code = LANGUAGE_MAP.get(language, "en-US")

    try:
        spoken_text = streamlit_js_eval(
            js_expressions=f"""
            new Promise((resolve) => {{
                console.log("Starting speech recognition...");
                const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.lang = '{language_code}';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;

                recognition.onresult = (event) => {{
                    const transcript = event.results[0][0].transcript;
                    console.log("Speech detected:", transcript);
                    resolve(transcript);  // Ensure this resolves the text
                }};
                
                recognition.onerror = (event) => {{
                    console.error("Speech Recognition Error:", event.error);
                    resolve("");  // Return an empty string on error
                }};
                
                recognition.onspeechend = () => {{
                    console.log("Speech ended.");
                    recognition.stop();
                }};
                
                recognition.start();
            }})
            """,
            want_output=True
        )

        if spoken_text and spoken_text.strip():  
            return spoken_text.strip()
        else:
            return None  

    except Exception as e:
        print(f"⚠️ Speech recognition failed: {e}")
        return None  


def translate_to_english(text, source_lang, detect_only=False):
    """Translate text to English or detect language."""
    if source_lang == "English":  
        return text  # No need to translate

    try:
        translator = Translator()

        # Detect language (useful if detect_only=True)
        detected_lang = translator.detect(text).lang if text.strip() else "en"
        normalized_lang = detected_lang.split('-')[0]  # Normalize (e.g., "zh-TW" → "zh")

        language_map = {
            "ms": "Malay",
            "zh": "Chinese",
            "ta": "Tamil"
        }

        if detect_only:
            return language_map.get(normalized_lang, "English")  # Return detected language name

        # Translate text
        translated_text = translator.translate(text, src=TRANSLATE_MAP.get(source_lang, "en"), dest="en").text
        return translated_text if translated_text else text  # If translation fails, return original text

    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return text  # Return original text on failure
 

"""
# language support code
LANGUAGE_MAP = {
    "English": "en",
    "Chinese": "zh-CN",
    "Malay": "ms-MY",
    "Tamil": "ta-IN"
}

TRANSLATE_MAP = {
    "Chinese": "zh-cn",
    "Malay": "ms",
    "Tamil": "ta"
}
from streamlit_js_eval import streamlit_js_eval

def speech_to_text():
    spoken_text = streamlit_js_eval(
        js_expressions=
        new Promise((resolve) => { 
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)(); 
            recognition.lang = 'en-US'; 
            recognition.interimResults = false; 
            recognition.maxAlternatives = 1; 
            recognition.continuous = false; 
            recognition.onresult = (event) => resolve(event.results[0][0].transcript); 
            recognition.onerror = (event) => resolve(""); 
            recognition.onspeechend = () => recognition.stop(); 
            recognition.start(); 
        })
        want_output=True
    )

    return spoken_text if spoken_text else "Speech recognition not supported"

# Translate function with detection option
def translate_to_english(text, source_lang, detect_only=False):
    if source_lang == "English":  # No translation needed
        return text

    try:
        translator = Translator()
        detected_lang = translator.detect(text).lang
        normalized_lang = detected_lang.split('-')[0]  # Get primary language code

        # Map normalized detected languages
        language_map = {
            "ms": "Malay",
            "id": "Malay",  # Treat Indonesian as Malay
            "zh": "Chinese",
            "ta": "Tamil"
        }

        if detect_only:
            return language_map.get(normalized_lang, "English")

        # Translate if detection is correct
        translated_text = translator.translate(
            text, 
            src=TRANSLATE_MAP.get(source_lang, "en"), 
            dest="en"
        ).text
        return translated_text

    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Return original if translation fails



main
#using google speech recognition
def speech_to_text(language='en', timeout=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio, language=LANGUAGE_MAP.get(language, "en"))
            print(f"Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again.")
            return None  
        except sr.RequestError:
            st.error("Speech Recognition service unavailable.")
            return None  
    
    return None

def translate_to_english(text, source_lang, detect_only=False):
    if source_lang == "English":  # No translation needed
        return text
    
    try:
        translator = Translator()
        
        # Detect language if detect_only is True
        detected_lang = translator.detect(text).lang
        
        # Normalize detected language for consistency
        normalized_lang = detected_lang.split('-')[0]  # Get primary code (e.g., ms from ms-MY)
        
        # Map normalized language codes to source languages
        language_map = {
            "ms": "Malay",
            "id": "Malay",  # Treat Indonesian as Malay
            "zh": "Chinese",
            "ta": "Tamil"
        }
        
        if detect_only:
            return language_map.get(normalized_lang, "English")
        
        # Translate if detect_only is False
        translated_text = translator.translate(
            text, 
            src=TRANSLATE_MAP.get(source_lang, "en"), 
            dest="en"
        ).text
        return translated_text
    
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Return original text if translation fails
"""
# Categorize issue using OpenAI
def categorize_issue(description):
    client = openai.Client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes facility issues."},
            {"role": "user", "content": f"Categorize this issue: {description}. Options: electrical, plumbing, other. Respond only with one of these options."}
        ]
    )
    return response.choices[0].message.content.strip().lower()

def check_ticket_status_by_user(user_identifier):
    tickets = session.query(Ticket).filter_by(user_identifier=user_identifier).all()
    
    if not tickets:
        return "No tickets found for this user."

    status_message = f"### Tickets for User: {user_identifier}\n\n"

    for ticket in tickets:
        status_message += f"🔹 **Ticket ID:** {ticket.ticket_id} (Status: {ticket.status})\n"
        status_message += f"📅 **Created At:** {ticket.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        issues = session.query(Issue).filter_by(ticket_id=ticket.ticket_id).all()
        
        if issues:
            for issue in issues:
                status = "Resolved ✅" if issue.resolved else "In Progress 🔄"
                status_message += f"  - **Issue Type:** {issue.type.capitalize()}\n"
                status_message += f"    **Description:** {issue.description}\n"
                status_message += f"    **Reported At:** {issue.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                status_message += f"    **Status:** {status}\n"
                
                assigned_worker = session.query(Worker).filter_by(worker_id=issue.assigned_worker_id).first()
                
                if assigned_worker:
                    status_message += f"👷 **Assigned Worker:** {assigned_worker.name}\n"
                else:
                    status_message += f"⚠ **No worker assigned yet**\n"
        
        status_message += "-------------------------------------\n"

    return status_message

#resolving for admin
def mark_issue_resolved(issue_id): 
    # Fetch the issue to be resolved
    issue = session.query(Issue).filter_by(issue_id=issue_id, resolved=False).first()
    
    if not issue:
        return "Issue not found or already resolved."

    # Mark issue as resolved
    issue.resolved = True
    session.commit()

    # Free the assigned worker
    if issue.assigned_worker_id:
        worker = session.query(Worker).filter_by(worker_id=issue.assigned_worker_id).first()
        if worker:
            # Check if the worker has other unresolved issues
            other_pending_issues = (
                session.query(Issue)
                .filter(
                    Issue.assigned_worker_id == worker.worker_id,
                    Issue.resolved == False
                )
                .count()
            )
            
            if other_pending_issues == 0:
                # Make worker available if no other unresolved issues
                worker.availability = True
                session.commit()

                # Auto-assign to the oldest unassigned issue
                pending_issue = (
                    session.query(Issue)
                    .join(Ticket)
                    .filter(
                        Issue.type == worker.specialization,
                        Issue.assigned_worker_id == None,
                        Ticket.status == "open"
                    )
                    .order_by(Issue.created_at)
                    .first()
                )

                # Assign the freed worker to the pending issue
                if pending_issue:
                    pending_issue.assigned_worker_id = worker.worker_id
                    worker.availability = False  # Worker is now busy again
                    session.commit()
                    return f"Issue {issue_id} resolved! Worker {worker.name} reassigned to Issue {pending_issue.issue_id}."
            
            # If other pending issues exist, worker remains unavailable
            else:
                return f"Issue {issue_id} resolved! Worker {worker.name} still has other pending issues."
    
    session.commit()
    return f"Issue {issue_id} resolved!"
