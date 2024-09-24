# AIEndoEnforcer

AIEndoEnforcer is an advanced AI-powered workout assistant and personal trainer application. It utilizes computer vision technology to track exercises in real-time, providing instant feedback and motivation through an AI chatbot.

## Features

- Real-time exercise tracking for squats and push-ups using computer vision
- AI-powered chatbot for personalized workout advice and motivation
- Streamlit-based user interface for easy interaction
- Integration with Google's Gemini AI for natural language processing
- Audio feedback using text-to-speech technology
- Comprehensive workout analysis and progress tracking

## Technologies Used

- Python
- OpenCV for computer vision
- MediaPipe for pose estimation
- Streamlit for the user interface
- Google Generative AI (Gemini) for natural language processing
- pyttsx3 for text-to-speech functionality

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AIEndoEnforcer.git
   cd AIEndoEnforcer
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   - Obtain a Google API key from the [Google Cloud Console](https://console.cloud.google.com/)
   - Set the environment variable:

     ```bash
     export GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the interface to start a workout session, interact with the AI trainer, and track your exercises.

## Project Structure

- `app.py`: Main application file containing the Streamlit UI and core logic
- `video_processor.py`: Handles video processing, pose estimation, and workout analysis
- `utils/pose_estimation.py`: Contains utility functions for pose estimation
- `Models/pose_landmarker_lite.task`: Pre-trained model for pose estimation

## Future Enhancements

- Support for additional exercise types
- Integration with wearable devices for more accurate tracking
- Personalized workout plans based on user progress and goals
- Social features for sharing achievements and competing with friends

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- Google Gemini AI for powering the chatbot functionality
- MediaPipe for pose estimation capabilities
- Streamlit for the user interface framework
