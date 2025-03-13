import openai
import streamlit as st
from PIL import Image
import io
from fuzzywuzzy import fuzz, process
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.orm import Session
from datetime import datetime
import speech_recognition as sr
import uuid
from dotenv import load_dotenv
import os
import cv2
import re 
import base64
import numpy as np
from numpy.linalg import norm
from googletrans import Translator 

# environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    resolved = Column(Boolean, default=False)  

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
            Worker(name="A", specialization="electrical", availability=True),
            Worker(name="B", specialization="electrical", availability=True),
            Worker(name="C", specialization="plumbing", availability=True),
            Worker(name="D", specialization="plumbing", availability=True),
            Worker(name="Management", specialization="other", availability=True)
        ]
        session.add_all(workers)
        session.commit()

initialize_workers()
#############################################################################################################
# get worker issues
def get_worker_issues(worker_id):
    pending_issues = session.query(Issue).filter_by(assigned_worker_id=worker_id, resolved=False).all()
    completed_issues = session.query(Issue).filter_by(assigned_worker_id=worker_id, resolved=True).all()
    return pending_issues, completed_issues

# Assign Worker to an Issue
def assign_worker(issue_type, user_identifier):
    
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

# Categorize issue using OpenAI
def categorize_issue(description):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes facility issues."},
            {"role": "user", "content": f"Categorize this issue: {description}. Options: electrical, plumbing, other. Respond only with one of these options."}
        ]
    )
    return response['choices'][0]['message']['content'].strip().lower()


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
