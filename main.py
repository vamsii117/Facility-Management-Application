import streamlit as st
from backend import (
    check_ticket_status_by_user, assign_worker, categorize_issue, mark_issue_resolved,extract_text_from_image,encode_image,fetch_issues,fetch_issue_data,
    speech_to_text, get_common_issues,identify_company_by_image,get_image_embedding,identify_ac_company,translate_to_english, session, Issue, Worker, Ticket, get_worker_issues, resolve_worker_issue
)
import uuid
import re
import plotly.express as px
import io
from PIL import Image
def report_issue():
    st.subheader("Report an Issue")
    user_identifier = st.text_input("Enter your username (in English):", key="report_user")
    language = st.selectbox("Choose your language:", ["English", "Malay", "Chinese", "Tamil"], key="language_select")

    # Initialize session state for issue description
    

    # ✅ Ensure session state is initialized
    if "issue_description" not in st.session_state:
        st.session_state.issue_description = ""

    # 🎤 Speech-to-Text Button
    if st.button("🎤 Speak instead"):
        st.info("🎙️ Listening...")  
        spoken_text = speech_to_text(language)  # Capture speech

        if spoken_text:  
            st.success(f"✅ Speech Recognized: {spoken_text}")  # Show success
        else:
            st.warning("⚠️ No speech detected. Please try again.")

    # 📝 Text Area with Persistent Speech Output
    issue_description = st.text_area(
        "Describe your issue:", 
        value=st.session_state.issue_description, 
        key="issue_description_input"
    )

    # Language validation
    if issue_description and language != "English":
        detected_language = translate_to_english(issue_description, language, detect_only=True)
        if detected_language != language:
            st.error(f"Please provide the input in the chosen language: {language}")
            return

    if st.button("Submit Issue", key="submit_issue_button"):
        if user_identifier and issue_description:
            translated_issue = translate_to_english(issue_description, language)  # Translate issue to English
            issue_type = categorize_issue(translated_issue)  # Categorize issue
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
            assigned_worker_id, assigned_worker_name = assign_worker(issue_type, user_identifier, translated_issue)

            # Create a new issue record
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


"""
# User reporting an issue.
def report_issue():
    st.subheader("Report an Issue")
    user_identifier = st.text_input("Enter your username (in English):", key="report_user")
    language = st.selectbox("Choose your language:", ["English", "Malay", "Chinese", "Tamil"], key="language_select")
    
    if "issue_description" not in st.session_state:
        st.session_state.issue_description = ""

    if st.button("🎤 Speak instead"):
        st.write("Listening...")
        spoken_text = speech_to_text(language)
        if spoken_text:
            st.session_state.issue_description = spoken_text
            st.rerun()  # Force UI update
        else:
            st.warning("No speech detected. Please try again.")

    issue_description = st.text_area("Describe your issue:", value=st.session_state.issue_description, key="issue_description_input")

    # Language validation
    if issue_description and language != "English":
        detected_language = translate_to_english(issue_description, language, detect_only=True)
        print(f"User Input: {issue_description}")  # Debugging
        print(f"Detected Language: {detected_language}, Selected: {language}") 
        if detected_language != language:
            st.error(f"Please provide the input in the chosen language: {language}")
            return
    #comment
    if st.button("🎤 Speak instead"): # for the speech to text implementation
        spoken_text = speech_to_text(language)
        if spoken_text:
            st.session_state.issue_description = spoken_text
        else:
            st.warning("No speech detected. Please try again.")
    issue_description = st.text_area("Describe your issue:", value=st.session_state.issue_description, key="issue_description_input")
    #comment
    #language validation new
    if issue_description and language != "English":  
        detected_language = translate_to_english(issue_description, language, detect_only=True)
        print(f"User Input: {issue_description}")
        print(f"Detected Language: {detected_language}, Selected: {language}")  # Debugging
        if detected_language != language:
            st.error(f"Please provide the input in the chosen language: {language}")
            return  
    #comment
    # language validation old
    if issue_description and language != "English":  
        detected_language = translate_to_english(issue_description, language, detect_only=True)
        if detected_language != language:
            st.error(f"Please provide the input in the chosen language: {language}")
            return  
    #comment
    if st.button("Submit Issue", key="submit_issue_button"):
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
            assigned_worker_id, assigned_worker_name = assign_worker(issue_type, user_identifier, translated_issue)

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
"""
def check_ticket_status():
    st.subheader("Check Ticket Status")
    user_identifier_status = st.text_input("Enter your Username/Email to check status:")
    if st.button("Check Status") and user_identifier_status:
        status_response = check_ticket_status_by_user(user_identifier_status)
        st.write(status_response)

def resolve_issue():
    st.title("Admin Dashboard")
    # Admin Login
    admin_password = st.text_input("Enter admin password:", type="password")
    if admin_password == "admin":  # Set a proper authentication method in production
        st.success("Access granted ")

        # Fetch issues
        df = fetch_issues()

        # **📌 Pending Issues**
        st.subheader("Pending Issues")
        pending_issues = df[df["status"] == "pending"]
        
        if pending_issues.empty:
            st.write("No pending issues.")
        else:
            for _, issue in pending_issues.iterrows():
                st.write(f"🔴 **Issue ID:** {issue['issue_id']}")
                st.write(f"**Type:** {issue['type'].capitalize()}")
                st.write(f"**Description:** {issue['description']}")
                st.write(f"**Reported At:** {issue['created_at']}")
                st.write("---")

        # **✅ Completed Issues**
        st.subheader("Completed Issues")
        completed_issues = df[df["status"] == "completed"]
        
        if completed_issues.empty:
            st.write("No completed issues.")
        else:
            for _, issue in completed_issues.iterrows():
                st.write(f"✅ **Issue ID:** {issue['issue_id']}")
                st.write(f"**Type:** {issue['type'].capitalize()}")
                st.write(f"**Description:** {issue['description']}")
                st.write("---")

        # **🔧 Resolve an Issue**
        st.subheader("Resolve a Specific Issue")
        issue_list = pending_issues["issue_id"].tolist()

        if issue_list:
            issue_to_resolve = st.selectbox("Select an Issue ID to mark as resolved:", issue_list)

            if st.button("Resolve Issue"):
                response = mark_issue_resolved(issue_to_resolve)
                st.success(response)
                st.rerun()  # Refresh the page after resolving

        else:
            st.info("No pending issues to resolve.")

        # **📊 Issues Tracking Graph**
        st.subheader("Daily Issue Statistics")

        df = fetch_issue_data()

        if df.empty:
            st.warning("No issue data available.")
            return

    # **📌 Bar Chart for Clear Day-wise View**
        df_melted = df.melt(id_vars=["date"], value_vars=["raised", "completed", "pending"],
                        var_name="Status", value_name="Count")

        fig = px.bar(df_melted, 
                 x="date", 
                 y="Count", 
                 color="Status", 
                 barmode="group", 
                 labels={"Count": "Number of Issues", "date": "Date"},
                 title="Day-wise Issue Tracking")

        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Issues", legend_title="Issue Status")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Invalid admin password! 🚫")
    
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

import streamlit as st
import openai
import base64
from PIL import Image
import io
import numpy as np
def report_issue_by_image():
    st.subheader("Report AC Issue by Image")

    uploaded_file = st.file_uploader("Upload an image of the AC", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        if "last_uploaded_file" in st.session_state and st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.detected_appliance = None
            st.session_state.issues = []
            st.session_state.selected_issue = None
            st.session_state.done_clicked = False

        st.session_state.last_uploaded_file = uploaded_file

        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Appliance"):
            img_bytes = io.BytesIO()
            uploaded_image.save(img_bytes, format="PNG")
            extracted_text = extract_text_from_image(img_bytes.getvalue())

            appliance = identify_ac_company(extracted_text)

            st.session_state.detected_appliance = appliance if appliance else "Unknown"
            st.session_state.issues = get_common_issues(appliance) if appliance else []
            st.session_state.selected_issue = None  # Reset issue selection

        if "detected_appliance" in st.session_state and st.session_state.detected_appliance:
            appliance = st.session_state.detected_appliance
            issues = st.session_state.issues

            if appliance != "Unknown":
                st.success(f"Detected AC Brand: {appliance}")

                if issues:
                    if st.session_state.selected_issue not in issues:
                        st.session_state.selected_issue = issues[0]

                    selected_issue_full = st.selectbox(
                        "Select your issue:",
                        options=issues,
                        index=issues.index(st.session_state.selected_issue) if st.session_state.selected_issue in issues else 0
                    )

                    # ✅ Extract only the issue name before storing
                    selected_issue_name = selected_issue_full.split(" - ")[0]

                    if st.button("Done"):
                        st.session_state.done_clicked = True
                        st.session_state.selected_issue = selected_issue_name

                if st.session_state.get("done_clicked", False):
                    issue_text = f"{appliance} AC - {st.session_state.selected_issue}"
                    st.session_state.issue_description = issue_text
                    st.info("You can now submit the issue.")

            else:
                st.error("Could not identify the AC brand. Please try another image.")

"""
def report_issue_by_image():
    st.subheader("Report AC Issue by Image")

    uploaded_file = st.file_uploader("Upload an image of the AC", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        if "last_uploaded_file" in st.session_state and st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.detected_appliance = None
            st.session_state.issues = []
            st.session_state.selected_issue = None
            st.session_state.done_clicked = False

        st.session_state.last_uploaded_file = uploaded_file

        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Appliance"):
            img_bytes = io.BytesIO()
            uploaded_image.save(img_bytes, format="PNG")
            extracted_text = extract_text_from_image(img_bytes.getvalue())

            appliance = identify_ac_company(extracted_text)

            st.session_state.detected_appliance = appliance if appliance else "Unknown"
            st.session_state.issues = get_common_issues(appliance) if appliance else []
            st.session_state.selected_issue = None  # Reset issue selection

        if "detected_appliance" in st.session_state and st.session_state.detected_appliance:
            appliance = st.session_state.detected_appliance
            issues = st.session_state.issues

            if appliance != "Unknown":
                st.success(f"Detected AC Brand: {appliance}")

                if issues:
                    if st.session_state.selected_issue not in issues:
                        st.session_state.selected_issue = issues[0]

                    selected_issue_full = st.selectbox(
                        "Select your issue:",
                        options=issues,
                        index=issues.index(st.session_state.selected_issue) if st.session_state.selected_issue in issues else 0
                    )

                    # ✅ Extract only the issue name before storing
                    selected_issue_name = selected_issue_full.split(" - ")[0]

                    if st.button("Done"):
                        st.session_state.done_clicked = True
                        st.session_state.selected_issue = selected_issue_name

                if st.session_state.get("done_clicked", False):
                    issue_text = f"{appliance} AC - {st.session_state.selected_issue}"
                    st.session_state.issue_description = issue_text
                    st.info("You can now submit the issue.")

            else:
                st.error("Could not identify the AC brand. Please try another image.")
"""
def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["User Section", "Report Issue by Image", "Admin Section", "Worker Dashboard"])
    
    if option == "User Section":
        report_issue()
        check_ticket_status()
    elif option == "Report Issue by Image":
        report_issue_by_image()
    elif option == "Admin Section":
        resolve_issue()
    elif option == "Worker Dashboard":
        worker_dashboard()
    

if __name__ == "__main__":
    main()
