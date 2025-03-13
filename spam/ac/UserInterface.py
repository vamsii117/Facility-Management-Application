import streamlit as st
from backend import (
    get_worker_issues,assign_worker,resolve_worker_issue,speech_to_text,translate_to_english,
    categorize_issue,check_ticket_status_by_user,mark_issue_resolved, session, Issue, Worker, Ticket
)
import uuid
import re

# User reporting an issue.
def report_issue():
    st.subheader("Report an Issue")
    user_identifier = st.text_input("Enter your username (in English):", key="report_user")
    
    language = st.selectbox("Choose your language:", ["English", "Malay", "Chinese", "Tamil"], key="language_select")
    
    if "issue_description" not in st.session_state: # used to store data througout an interaction span
        st.session_state.issue_description = ""
    
    if st.button("🎤 Speak instead"): # for the speech to text implementation
        spoken_text = speech_to_text(language)
        if spoken_text:
            st.session_state.issue_description = spoken_text
        else:
            st.warning("No speech detected. Please try again.")
    
    issue_description = st.text_area("Describe your issue:", value=st.session_state.issue_description, key="issue_description_input")
    
    # language validation
    if issue_description and language != "English":  
        detected_language = translate_to_english(issue_description, language, detect_only=True)
        if detected_language != language:
            st.error(f"Please provide the input in the chosen language: {language}")
            return  

    if st.button("Submit Issue"):
        if user_identifier and issue_description:
            translated_issue = translate_to_english(issue_description, language)  # Translate issue to English
            issue_type = categorize_issue(translated_issue)  # Categorize in using open ai
            issue_type = re.sub(r'[^a-zA-Z]', '', issue_type).strip().lower()  

            st.write(f"Issue categorized as: {issue_type}")
            
            # Check for existing ticket or create a new one
            existing_ticket = session.query(Ticket).filter_by(user_identifier=user_identifier, status='open').first()
            if existing_ticket:
                ticket_id = existing_ticket.ticket_id
            else:
                ticket_id = str(uuid.uuid4())
                new_ticket = Ticket(ticket_id=ticket_id, user_identifier=user_identifier)
                session.add(new_ticket)
                session.commit()

            # Assign worker based on issue type and user
            assigned_worker_id, assigned_worker_name = assign_worker(issue_type, user_identifier)

            # Create a new issue record to store decription in english
            new_issue = Issue(
                ticket_id=ticket_id,
                type=issue_type,
                description=translated_issue,
                assigned_worker_id=assigned_worker_id
            )
            session.add(new_issue)
            session.commit()

            st.write(f"Assigned Worker: {assigned_worker_name}")
        else:
            st.error("Please fill in all fields.")

def check_ticket_status():
    st.subheader("Check Ticket Status")
    user_identifier_status = st.text_input("Enter your Username/Email to check status:")
    if st.button("Check Status") and user_identifier_status:
        status_response = check_ticket_status_by_user(user_identifier_status)
        st.write(status_response)

def resolve_issue():
    st.subheader("Resolve an Issue")
    admin_password = st.text_input("Enter admin password:", type="password")
    
    if admin_password == "admin":  # password
        # Display Pending Issues
        st.subheader("Pending Issues")
        pending_issues = session.query(Issue).filter_by(resolved=False).all()
        
        if not pending_issues:
            st.write("No pending issues.")
        else:
            for issue in pending_issues:
                st.write(f"🔴 **Issue ID:** {issue.issue_id}")
                st.write(f"**Type:** {issue.type.capitalize()}")
                st.write(f"**Description:** {issue.description}")
                st.write(f"**Reported At:** {issue.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write("---")

        # Display Completed Issues
        st.subheader("Completed Issues")
        completed_issues = session.query(Issue).filter_by(resolved=True).all()
        
        if not completed_issues:
            st.write("No completed issues.")
        else:
            for issue in completed_issues:
                st.write(f"✅ **Issue ID:** {issue.issue_id}")
                st.write(f"**Type:** {issue.type.capitalize()}")
                st.write(f"**Description:** {issue.description}")

        # Allow Admin to Enter Issue ID for Resolution
        st.subheader("Resolve a Specific Issue")
        issue_to_resolve = st.text_input("Enter Issue ID to mark as resolved:")
        if st.button("Resolve Issue"):
            if issue_to_resolve:
                response = mark_issue_resolved(issue_to_resolve)
                st.success(response)
                st.rerun()  # Refresh the page to update lists
            else:
                st.error("Please enter a valid Issue ID.")
    else:
        st.error("Invalid admin password!")

def worker_dashboard():
    st.subheader("Worker Dashboard")
    worker_name = st.text_input("Enter your name to log in:")

    if worker_name:
        worker = session.query(Worker).filter_by(name=worker_name).first()
        
        if worker:
            st.write(f"Welcome, {worker_name}!")
            pending_issues, completed_issues = get_worker_issues(worker.worker_id)

            st.subheader("Pending Issues")
            if not pending_issues:
                st.write("No pending issues assigned.")
            else:
                for issue in pending_issues:
                    st.write(f"**Issue ID:** {issue.issue_id}")
                    st.write(f"**Type:** {issue.type.capitalize()}")
                    st.write(f"**Description:** {issue.description}")
                    st.write(f"**Reported At:** {issue.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if st.button(f"Mark as Completed (Issue {issue.issue_id})", key=f"resolve_{issue.issue_id}"):
                        response = resolve_worker_issue(issue.issue_id, worker.worker_id)
                        st.success(response)
                        st.rerun()  # to refresh the page

            st.subheader("Completed Issues")
            if not completed_issues:
                st.write("No completed issues.")
            else:
                for issue in completed_issues:
                    st.write(f"✅ **Issue ID:** {issue.issue_id} - {issue.type.capitalize()} - Resolved")

        else:
            st.error("Worker not found. Please enter a valid name.")

def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["User Section", "Admin Section", "Worker Dashboard"])
    
    if option == "User Section":
        report_issue()
        check_ticket_status()
    elif option == "Admin Section":
        resolve_issue()
    elif option == "Worker Dashboard":
        worker_dashboard()

if __name__ == "__main__":
    main()
