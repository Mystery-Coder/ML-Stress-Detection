// Variables 
let currentQuestion = 0;
const questions = [
    "how are you doing today",
    "where are you from originally",
    "what are some things you really like about the place ", 
    "How easy was it for you to get used to living in the place where you live?",
    "What are some things you don't really like about the place where you live?",
    "what'd you study at school",
    "are you still doing that",
    "what's your dream job",
    "do you travel a lot?",
    "How often do you go back to your hometown?",
    "Do you consider yourself an introvert ?",
    "What do you do to relax ?",
    "How are you at controlling your temper?",
    "When was the last time you argued with someone and what was it about ?",
    "When was the last time you argued with someone, and what was it about? How did you feel in that moment?",
    "Tell me more about that argument?",
    "How close are you to that person?",
    "How do you know them ?",
    "What are some things you like to do for fun ?",
    "Who's someone that's been a positive influence in your life ?",
    "Can you tell me more about them?",
    "How close are you to your family",
    "What made you decide to take the course?",
    "Could you have done anything differently to avoid it?",
    "What's one of your most memorable experiences ?",
    "How was your college life?",
    "How do you like your current  living situation?",
    "How easy is it for you to get a good night's sleep?",
    "Do you feel that way often ?",
    "What are you like when you don't sleep well?",
    "what are you like when you don't sleep well",
    "Have you ever felt down before?",
    "Have you been diagnosed with depression ?",
    "Have you ever been diagnosed with p_t_s_d",
    "have you ever served in the military",
    "When was the last time you felt really happy?",
    "Tell me more about that ?",
    "What do you think of today's kids?",
    "What do you do when you're annoyed ?",
    "When was the last time that happened ?",
    "Can you tell me about that ?",
    "How would your best friend describe you ?",

];

let recorder, audioBlob, audioURL, timerInterval, audioSeconds = 0;
let audio;  // Global audio variable to control playback
let progressBar = document.getElementById('progress-bar');

// Get DOM elements
const questionElement = document.getElementById("question");
const timerElement = document.getElementById("audio-timer");
const playButton = document.getElementById("play-audio");
const stopButton = document.getElementById("stop-recording");
const startButton = document.getElementById("start-recording");
const nextButton = document.getElementById("next-question");
const submitButton = document.getElementById("submit-button");
const flashMessage = document.getElementById("flash-message");
const stopAudioButton = document.getElementById("stop-audio");
const responsesForm = document.getElementById("responsesForm");
const spinner = document.createElement('div');
spinner.classList.add('spinner');
flashMessage.appendChild(spinner);

// Audio responses map
const audioResponses = new Map();

// Update Question
function updateQuestion() {
    questionElement.textContent = `Question ${currentQuestion + 1}: ${questions[currentQuestion]}`;
    // Hide Submit button until the last question
    if (currentQuestion === questions.length - 1) {
        submitButton.style.display = "block";
        nextButton.style.display = "none";  // Hide Next button on the last question
    } else {
        submitButton.style.display = "none";
        nextButton.style.display = "block";
    }
}

// Format time for the timer
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return` ${mins}:${secs < 10 ? "0" : ""}${secs}`;
}

// Start Recording
startButton.addEventListener("click", () => {
    startButton.disabled = true;  // Disable Start button
    stopButton.disabled = false;  // Enable Stop button
    playButton.disabled = true;  // Disable Play button until recording is done
    audioSeconds = 0;
    timerElement.textContent = "0:00";
    progressBar.style.width = "0%";  // Reset progress bar when starting recording

    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        recorder = new MediaRecorder(stream);
        recorder.start();

        // Timer
        timerInterval = setInterval(() => {
            audioSeconds++;
            timerElement.textContent = formatTime(audioSeconds);
        }, 1000);

        recorder.addEventListener("dataavailable", (event) => {
            audioBlob = event.data;
            audioURL = URL.createObjectURL(audioBlob);
            playButton.disabled = false; // Enable play button after recording
            // Save audio blob to the current question
            audioResponses.set(currentQuestion, audioBlob);
        });
    }).catch((error) => {
        console.error("Error accessing the microphone: ", error);
    });
});

// Stop Recording
stopButton.addEventListener("click", () => {
    stopButton.disabled = true;  // Disable Stop button once clicked
    startButton.disabled = false; // Enable Start button for next recording

    recorder.stop();  // Stop the recording
    clearInterval(timerInterval); // Stop the timer
});

// Play Audio
playButton.addEventListener("click", () => {
    if (audio) {
        audio.pause();
        audio.currentTime = 0; // Reset the audio to the start
    }
    audio = new Audio(audioURL);
    audio.play();

    // Reset progress bar when audio starts playing
    progressBar.style.width = "0%";

    audio.addEventListener('timeupdate', function () {
        const progress = (audio.currentTime / audio.duration) * 100;
        progressBar.style.width = progress + "%";
    });

    // Enable Stop Audio button when audio starts playing
    stopAudioButton.disabled = false;

    // Ensure play button is disabled after playback is done
    audio.onended = function() {
        playButton.disabled = true;  // Disable Play button once audio ends
        stopAudioButton.disabled = true;  // Disable Stop Audio button when audio finishes
    };
});

// Stop Audio
stopAudioButton.addEventListener("click", () => {
    if (audio) {
        audio.pause();  // Stop the audio playback
        audio.currentTime = 0;  // Reset audio position to the start
        progressBar.style.width = "0%";  // Reset progress bar
    }
    stopAudioButton.disabled = true;  // Disable Stop button after stopping
    playButton.disabled = false;  // Enable play button to allow replay
});

// Navigation
nextButton.addEventListener("click", () => {
    if (currentQuestion < questions.length - 1) {
        currentQuestion++;
        updateQuestion();
        resetRecordingState(); // Reset recording states for the next question
    }
});

// Submit Action
submitButton.addEventListener("click", async (event) => {
    event.preventDefault(); // Prevent form default submission

    // Show a loading spinner or flash message
    flashMessage.style.display = "block";
    flashMessage.innerHTML = "<div class='spinner'></div> Processing... Please wait.";

    // Prepare the data to send to the server
    const testData = {
        answers: questions.map((_, index) => {
            return {
                question: questions[index],
                answer: document.getElementById(`answer-${index}).value || No answer`)
            };
        })
    };

    try {
        // Send the test data to the backend using axios
        const response = await axios.post('/submit_test', testData);

        // Check if the backend returns emotion analysis data
        if (response.status === 200) {
            const result = response.data.result;
            const emotionAnalysis = response.data.emotion_analysis; // Assuming the backend sends this data

            // Update the flash message with the result
            flashMessage.innerHTML = <div class="success-message">Test result: ${result}</div>;

            // Delay the display of emotion analysis by 5 seconds
            setTimeout(() => {
                const emotionAnalysisDiv = document.getElementById("emotion-analysis");

                // Ensure the emotion analysis section is visible
                emotionAnalysisDiv.style.display = "block";
                emotionAnalysisDiv.innerHTML = `
                    <h3>Emotion Analysis</h3>
                    <pre>${JSON.stringify(emotionAnalysis, null, 2)}</pre> <!-- Display JSON data nicely -->
                `;

                // Optionally, you could add additional logic to handle the analysis further (e.g., displaying charts)
            }, 5000); // Delay of 5000 milliseconds (5 seconds)
        } else {
            flashMessage.innerHTML = <div class="error-message">Error: Unable to process the test.</div>;
        }
    } catch (error) {
        // Handle errors
        console.error('Error submitting test:', error);
        flashMessage.innerHTML = <div class="error-message">An error occurred: ${error.message}</div>;
    }
});

// Reset all states when moving to the next question
function resetRecordingState() {
    // Reset UI elements for the next question
    stopButton.disabled = true;
    startButton.disabled = false;
    playButton.disabled = true;
    stopAudioButton.disabled = true; // Disable stop audio button when reset
    progressBar.style.width = "0%";
    timerElement.textContent = "0:00";
    audioSeconds = 0; 
}

// Initial question update
updateQuestion();