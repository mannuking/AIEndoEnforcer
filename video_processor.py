import cv2
import numpy as np
from utils.pose_estimation import process_frame_for_pose, analyze_pose, RepCounter, display_workout_stats
import streamlit as st
import mediapipe as mp
import time
import pyttsx3
import logging
import google.generativeai as genai
import os
import threading
import random

# Load environment variables
load_dotenv()

# Access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure logging
logging.basicConfig(
    filename="workout_app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

class VideoProcessor:
    def __init__(self):
        self.current_workout = None
        self.rep_counter = RepCounter()
        self.workout_start_time = None
        self.ai_feedback = ""
        self.analysis_count = 0
        self.workout_data = []
        self.reps_per_set = 20
        logging.info("VideoProcessor initialized.")

    def process_video(self):
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        st.write("Workout Started...")

        frame_placeholder = st.empty()
        stop_button_placeholder = st.empty()

        # Move the stop button outside the video processing loop
        if stop_button_placeholder.button("Stop Workout"):
            st.session_state.workout_active = False

        start_time = time.time()
        last_analysis_time = time.time()

        while st.session_state.workout_active:
            ret, frame = cap.read()
            if not ret:
                break

            pose_landmarks, annotated_frame = process_frame_for_pose(frame)

            if pose_landmarks:
                self.current_workout = st.session_state.current_exercise  # Get current exercise from session state

                if not self.workout_start_time:
                    self.workout_start_time = time.time()

                pose_data = analyze_pose(pose_landmarks, self.rep_counter)
                self.workout_data.append(pose_data)

                annotated_frame = self.display_simplified_stats(annotated_frame, pose_data)

                # Check if we should start real-time analysis
                total_reps = self.rep_counter.squat_count if self.current_workout == "Squats" else self.rep_counter.pushup_count
                if total_reps >= 5 and (not hasattr(self, "analysis_thread") or not self.analysis_thread.is_alive()):
                    # Start real-time analysis in a separate thread
                    self.analysis_thread = threading.Thread(target=self.run_realtime_analysis)
                    self.analysis_thread.start()

                # Check if the set is finished (20 reps)
                if total_reps >= self.reps_per_set:
                    if self.current_workout == "Squats":
                        # Transition to push-ups
                        st.write("Squats Finished! Get ready for Push-ups...")
                        self.current_workout = "Push-ups"
                        self.rep_counter = RepCounter()  # Reset the rep counter for push-ups
                        self.workout_start_time = time.time()  # Reset workout start time
                        self.workout_data = []  # Reset workout data
                    else:
                        # Workout finished
                        st.session_state.workout_active = False

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

        # Workout has stopped, perform final analysis
        cap.release()
        cv2.destroyAllWindows()
        st.write("Workout Finished")

        self.run_final_analysis()

    def run_realtime_analysis(self):
        if not self.workout_data:
            return

        # Generate short motivational prompt
        motivational_phrases = [
            "Keep going!", "Push harder!", "Believe in yourself!",
            "Don't stop now!", "Stay focused!", "Challenge yourself!",
            "You're unstoppable!", "Don't quit!", "Give it your all!",
            "Keep pushing!", "Rise above limits!", "Push past the pain!"
        ]

        # Prompt for real-time motivational feedback under 20 words
        prompt = f"You are a motivational fitness coach. Choose one motivational phrase from this list:\n{motivational_phrases}\n\nBased on the following real-time workout data:\n"

        data_to_analyze = {
            "total_reps": self.rep_counter.squat_count if self.current_workout == "Squats" else self.rep_counter.pushup_count,
            "total_sets": self.rep_counter.set_count,
            "average_left_leg_angle": np.mean([data['left_leg_angle'] for data in self.workout_data[-30:] if 'left_leg_angle' in data]),
            "average_left_arm_angle": np.mean([data['left_arm_angle'] for data in self.workout_data[-30:] if 'left_arm_angle' in data]),
        }
        prompt += str(data_to_analyze)

        response = model.generate_content(prompt)

        # Get only the motivational phrase under 20 words
        feedback = response.text.strip()[:20]

        # Display and speak the feedback
        st.write(feedback)
        self.provide_audio_feedback(feedback)


    def provide_audio_feedback(self, text):
        speech_thread = threading.Thread(target=self.speak_text, args=(text,))
        speech_thread.start()

    def speak_text(self, text):
        """Function to be executed in a separate thread for text-to-speech."""
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', tts_engine.getProperty('rate') * 1)  # Set speed to 0.9x
        tts_engine.say(text)
        tts_engine.runAndWait()

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def display_simplified_stats(self, image, stats):
        total_reps = stats['squat_count'] if self.current_workout == "Squats" else stats['pushup_count']
        image = draw_text_on_image(image, f"Reps: {total_reps}", (10, 30), font_scale=1.0)  # Increased font size
        # Removed set count display
        return image

    def display_feedback_cards(self, feedback_dict, max_cards=2):
        """Displays separate cards for Progress Motivation and Areas for Improvement,
        removing bold markdown (**).
        """
        for category, text in feedback_dict.items():
            # Generate a random color for the card
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 

            # Remove bold markdown from the text
            text = text.replace("**", "")

            # Create a container for the card
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: rgb({color[0]}, {color[1]}, {color[2]}); padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <p style="color: white; font-size: 18px;">{category}: {text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            self.provide_audio_feedback(text)

   
    def run_final_analysis(self):
        if not self.workout_data:
            st.write("No workout data to analyze.")
            return

        # Collect data for analysis 
        data_to_analyze = {  
            "total_reps": self.rep_counter.squat_count if self.current_workout == "Squats" else self.rep_counter.pushup_count,
            "total_sets": self.rep_counter.set_count,
            "average_left_leg_angle": np.mean([data['left_leg_angle'] for data in self.workout_data if 'left_leg_angle' in data]),
            "average_left_arm_angle": np.mean([data['left_arm_angle'] for data in self.workout_data if 'left_arm_angle' in data]),
            "workout_duration": time.time() - self.workout_start_time,
        }

        # Enhanced prompts for Progress Motivation and Areas for Improvement
        progress_prompt = f"""
        You are a highly motivating fitness coach. 
        Analyze the following complete {self.current_workout} data. 
        Provide a detailed and insightful summary of the user's progress, 
        highlighting their strengths and achievements. 
        Offer encouragement and motivation for future workouts.
        Ensure that your response is a few complete sentences but only in 30 words.

        Data: {str(data_to_analyze)}
        """

        improvement_prompt = f"""
        You are a knowledgeable and supportive fitness coach. 
        Analyze the following complete {self.current_workout} data. 
        Provide specific and actionable suggestions for improvement, 
        focusing on form, technique, and potential areas of weakness. 
        Be constructive and encouraging in your feedback.
        Make sure your response is a few complete sentences but only in 30 words.

        Data: {str(data_to_analyze)}
        """

        # Get responses from Gemini (no character limit)
        progress_response = model.generate_content(progress_prompt).text.strip()
        improvement_response = model.generate_content(improvement_prompt).text.strip()

        # Organize feedback into a dictionary
        feedback_dict = {
            "Progress": progress_response,
            "Areas for Improvement": improvement_response,
        }

        # Display the feedback cards
        self.display_feedback_cards(feedback_dict)

        self.workout_data = []  # Reset workout data for the next workout

        st.session_state.chatbot_active = True



def draw_text_on_image(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
                       color=(255, 255, 255), thickness=1):
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image
