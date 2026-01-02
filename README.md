# 🎤 ML Stress Detection System

A Flask-based web application that uses machine learning to detect stress levels and emotions from audio recordings. The system analyzes voice patterns to predict emotional states and depression levels through advanced audio processing and deep learning models.

## ✨ Features

-   **User Authentication**: Secure registration and login system with password hashing
-   **Audio Recording**: Record and upload audio responses directly through the web interface
-   **Multi-Step Audio Processing**:
    -   Automatic merging of multiple audio files
    -   Intelligent audio chunking (2-minute chunks for depression analysis, 3-second chunks for emotion detection)
-   **Machine Learning Predictions**:
    -   Emotion classification using TensorFlow/Keras models
    -   Depression level assessment
    -   MFCC (Mel-frequency cepstral coefficients) feature extraction
-   **Visual Analytics**: Interactive charts and graphs showing:
    -   Emotion distribution (bar chart)
    -   Emotion frequency over time (line chart)
    -   Depression levels (line chart)
    -   Emotion pie chart
-   **Real-time Processing**: Background task processing for efficient audio analysis

## 🛠️ Technology Stack

-   **Backend Framework**: Flask
-   **Machine Learning**: TensorFlow/Keras, scikit-learn
-   **Audio Processing**: librosa, pydub, soundfile
-   **Database**: SQLite with SQLAlchemy ORM
-   **Data Visualization**: Matplotlib
-   **Other Libraries**: NumPy, Joblib, Resampy

## 📋 Prerequisites

-   Python 3.10.11
-   pip (Python package manager)
-   Virtual environment (recommended)

## 📖 Usage

1. **Start the Flask application**

    ```bash
    python Webapp.py
    ```

2. **Access the web application**

    - Open your browser and navigate to: `http://127.0.0.1:5000`

3. **Register a new account**

    - Click on "Register" and create an account with username, email, and password

4. **Login**

    - Use your credentials to log in

5. **Take the test**

    - Click on "Take the Test"
    - Record audio responses to the questions
    - Submit your recordings

6. **View results**
    - After processing, view detailed results with charts and analysis
    - Results include emotion distribution, frequency over time, and depression levels

## 📁 Project Structure

```
ml-stress-detection/
│
├── Webapp.py                 # Main Flask application
├── requirements.txt          # Python dependencies
│
├── Model/                    # ML model files
│   ├── emotion.keras
│   └── depression.keras
│
├── lb/                       # Label encoders
│   ├── lb-emotion.sav
│   └── lb-depression.sav
│
├── templates/                # HTML templates
│   ├── InterfaceIndex.html
│   ├── login.html
│   ├── register.html
│   ├── take_test.html
│   ├── result.html
│   └── ...
│
├── static/                   # Static files
│   ├── css/                  # Stylesheets
│   ├── js/                   # JavaScript files
│   ├── images/               # Images
│   ├── graphs/               # Generated charts
│   └── uploads/              # Uploaded audio files
│
├── audiofile-step1/          # Merged audio files
├── audiofile-step3-2/        # 2-minute audio chunks (depression analysis)
├── audiofile-step3-3/        # 3-second audio chunks (emotion analysis)
│
├── step1-adder.py            # Audio merging script
├── step2-audio.py            # Audio splitting script
├── step3-Model.py            # ML prediction script
├── step4-graph.py            # Chart generation script
│
├── users_db.py               # Database utilities
├── run_models.py             # Model evaluation script
└── predictions.json          # Prediction results (generated)
```

## 🔄 How It Works

1. **Audio Upload**: Users record and upload audio files through the web interface
2. **Audio Merging**: Multiple audio files are merged into a single WAV file
3. **Audio Chunking**:
    - Audio is split into 2-minute chunks for depression analysis
    - Audio is split into 3-second chunks for emotion detection
4. **Feature Extraction**: MFCC features are extracted from each audio chunk
5. **ML Prediction**:
    - Emotion model predicts emotions for each 3-second chunk
    - Depression model predicts depression levels for each 2-minute chunk
6. **Visualization**: Results are visualized through multiple chart types
7. **Result Display**: Users view comprehensive analysis results with visualizations

## 🔒 Security Notes

-   **Development Server**: The application runs on Flask's development server (not suitable for production)
-   **Secret Key**: Change the default secret key in production environments
-   **Database**: Uses SQLite (consider PostgreSQL/MySQL for production)
-   **User Routes**: The `/view_users` route should be removed or protected in production

## ⚠️ Important Notes

-   Ensure all model files (`*.keras` and `*.sav`) are present before running
-   Audio files are processed in the background; larger files may take longer
-   The application requires sufficient disk space for audio processing
-   Supported audio formats: WAV, WebM

## 🐛 Troubleshooting

-   **Model Loading Errors**: Ensure model files are in the correct directories
-   **Audio Processing Errors**: Check that librosa and soundfile are properly installed
-   **Database Errors**: Run `/init_db` route to initialize database tables
-   **Port Conflicts**: Change the port in `Webapp.py` if port 5000 is already in use
