import streamlit as st
import os
from video_processor import VideoProcessor
import google.generativeai as genai


# --- Page Configuration ---
st.set_page_config(page_title="AIEndoEnforcer", page_icon=":muscle:")

# Initialize session state variables
if "workout_active" not in st.session_state:
    st.session_state.workout_active = False
if "video_processor" not in st.session_state:
    st.session_state.video_processor = None
if "current_exercise" not in st.session_state:
    st.session_state.current_exercise = None
if "chatbot_active" not in st.session_state:
    st.session_state.chatbot_active = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Page Title ---
st.title("AIEndoEnforcer")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# --- Workout Controls ---
if not st.session_state.workout_active:
    if st.button("Start Workout"):
        st.session_state.workout_active = True
        st.session_state.current_exercise = "Squats"
        st.session_state.video_processor = VideoProcessor()
else:
    if st.session_state.current_exercise == "Squats" and st.button("Finish Squats"):
        st.session_state.workout_active = False
        st.session_state.video_processor.run_final_analysis()
        st.session_state.current_exercise = "Push-ups"
        if st.button("Start Push-ups"):
            st.session_state.workout_active = True
            st.session_state.video_processor = VideoProcessor()
    elif st.session_state.current_exercise == "Push-ups" and st.button("Finish Push-ups"):
        st.session_state.workout_active = False
        st.session_state.video_processor.run_final_analysis()

# --- Chatbot ---
if st.session_state.chatbot_active:
    st.write("## Chat with Your AI Trainer:")
    user_input = st.text_input("You:", "")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Construct the prompt for Gemini
        prompt = f"""
        You are a highly dedicated and motivating AI personal trainer. 
        You are passionate about helping users achieve their fitness goals.
        Provide helpful, encouraging, and supportive responses.
        Offer insightful workout tips and advice.

        Chat History: {st.session_state.chat_history}

        User: {user_input}
        AI Trainer: 
        """

        # Generate response from Gemini
        response = model.generate_content(prompt)
        ai_response = response.text.strip()

        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"*You:* {message['content']}")
            else:
                st.write(f"*AI Trainer:* {message['content']}")

# --- Real-time Workout Display ---
if st.session_state.workout_active:
    video_processor = st.session_state.video_processor
    video_processor.process_video()
