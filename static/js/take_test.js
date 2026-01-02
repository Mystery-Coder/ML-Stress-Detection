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

let recorder = null;
let audioChunks = []; // ✅ Collect chunks during recording
let audioBlob = null;
let audioURL = null;
let timerInterval = null;
let audioSeconds = 0;
let audio = null; // Global audio variable to control playback
let mediaStream = null; // ✅ Store stream to stop tracks later

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
const progressBar = document.getElementById("progress-bar");
const debugButton = document.getElementById("debug-question");
// Audio responses map
const audioResponses = new Map();

// Update Question
function updateQuestion() {
	questionElement.textContent = `Question ${currentQuestion + 1}: ${
		questions[currentQuestion]
	}`;
	// Hide Submit button until the last question
	if (currentQuestion === questions.length - 1) {
		submitButton.style.display = "block";
		nextButton.style.display = "none";
	} else {
		submitButton.style.display = "none";
		nextButton.style.display = "block";
	}
}

// Format time for the timer
function formatTime(seconds) {
	const mins = Math.floor(seconds / 60);
	const secs = seconds % 60;
	return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
}

// Start Recording
startButton.addEventListener("click", async () => {
	console.log("Start recording clicked");

	// Reset state
	audioChunks = [];
	audioBlob = null;
	audioURL = null;
	audioSeconds = 0;
	timerElement.textContent = "0:00";
	progressBar.style.width = "0%";

	// Update button states
	startButton.disabled = true;
	stopButton.disabled = false;
	playButton.disabled = true;

	try {
		// Request microphone access
		mediaStream = await navigator.mediaDevices.getUserMedia({
			audio: true,
		});

		// ✅ Check browser support for WebM
		const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
			? "audio/webm;codecs=opus"
			: "audio/webm";

		console.log(`Using MIME type: ${mimeType}`);

		// Create MediaRecorder with proper configuration
		recorder = new MediaRecorder(mediaStream, {
			mimeType: mimeType,
			audioBitsPerSecond: 128000, // 128kbps for good quality
		});

		// ✅ Collect audio chunks as they're available
		recorder.ondataavailable = (event) => {
			if (event.data && event.data.size > 0) {
				console.log(`Chunk received: ${event.data.size} bytes`);
				audioChunks.push(event.data);
			}
		};

		// ✅ Handle recording stop
		recorder.onstop = () => {
			console.log("Recording stopped, creating blob...");

			// Create final blob from all chunks
			audioBlob = new Blob(audioChunks, { type: mimeType });
			console.log(`Final blob size: ${audioBlob.size} bytes`);

			// ✅ Validate blob
			if (audioBlob.size < 100) {
				console.error("Recording failed - blob too small!");
				alert("Recording failed. Please try again.");
				resetRecordingState();
				return;
			}

			// Create URL for playback
			audioURL = URL.createObjectURL(audioBlob);

			// Save to responses map
			audioResponses.set(currentQuestion, audioBlob);
			console.log(`Saved audio for question ${currentQuestion}`);

			// Enable play button
			playButton.disabled = false;

			// Stop all media tracks
			if (mediaStream) {
				mediaStream.getTracks().forEach((track) => track.stop());
			}
		};

		// ✅ Handle errors
		recorder.onerror = (event) => {
			console.error("Recording error:", event.error);
			alert("Recording error: " + event.error);
			resetRecordingState();
		};

		// Start recording (request data every 1 second for better chunk collection)
		recorder.start(1000);
		console.log("Recording started");

		// Start timer
		timerInterval = setInterval(() => {
			audioSeconds++;
			timerElement.textContent = formatTime(audioSeconds);
		}, 1000);
	} catch (error) {
		console.error("Error accessing microphone:", error);
		alert("Could not access microphone. Please check permissions.");
		resetRecordingState();
	}
});

// Stop Recording
stopButton.addEventListener("click", () => {
	console.log("Stop recording clicked");

	if (recorder && recorder.state !== "inactive") {
		stopButton.disabled = true;
		recorder.stop(); // This triggers onstop event
		clearInterval(timerInterval);
	}
});

// Play Audio
playButton.addEventListener("click", () => {
	if (!audioURL) {
		console.error("No audio to play");
		return;
	}

	// Stop any currently playing audio
	if (audio) {
		audio.pause();
		audio.currentTime = 0;
	}

	audio = new Audio(audioURL);
	progressBar.style.width = "0%";

	audio.play();
	console.log("Playing audio");

	// Update progress bar
	audio.addEventListener("timeupdate", function () {
		const progress = (audio.currentTime / audio.duration) * 100;
		progressBar.style.width = progress + "%";
	});

	// Enable/disable buttons based on playback
	stopAudioButton.disabled = false;
	playButton.disabled = true;

	audio.onended = function () {
		console.log("Audio playback finished");
		playButton.disabled = false;
		stopAudioButton.disabled = true;
		progressBar.style.width = "100%";
	};
});

// Stop Audio
stopAudioButton.addEventListener("click", () => {
	if (audio) {
		audio.pause();
		audio.currentTime = 0;
		progressBar.style.width = "0%";
	}
	stopAudioButton.disabled = true;
	playButton.disabled = false;
});

// Navigation
nextButton.addEventListener("click", () => {
	// ✅ Check if current question has a recording
	// if (!audioResponses.has(currentQuestion)) {
	// 	alert("Please record an answer before moving to the next question.");
	// 	return;
	// }

	if (currentQuestion < questions.length - 1) {
		currentQuestion++;
		updateQuestion();
		resetRecordingState();
	}
});

debugButton.addEventListener("click", () => {
	//For debug move to q 40

	if (currentQuestion < questions.length - 1) {
		currentQuestion = 39;
		updateQuestion();
		resetRecordingState();
	}
});

// Submit Action
submitButton.addEventListener("click", async (event) => {
	event.preventDefault();

	// ✅ Validate all questions have recordings
	// if (audioResponses.size !== questions.length) {
	// 	alert(
	// 		`Please answer all questions. You've answered ${audioResponses.size} out of ${questions.length}.`
	// 	);
	// 	return;
	// }

	// Show loading state
	flashMessage.style.display = "block";
	flashMessage.innerHTML =
		"<div class='spinner'></div> Processing... Please wait.";

	// Disable submit button to prevent double submission
	submitButton.disabled = true;

	console.log("Preparing form data...");
	const formData = new FormData();

	// ✅ Append audio files with validation
	let totalSize = 0;
	questions.forEach((question, index) => {
		if (audioResponses.has(index)) {
			const blob = audioResponses.get(index);
			console.log(`Adding question ${index}: ${blob.size} bytes`);
			totalSize += blob.size;
			formData.append(`audio-${index}`, blob, `question-${index}.webm`);
		}
	});

	console.log(
		`Total upload size: ${totalSize} bytes (${(
			totalSize /
			1024 /
			1024
		).toFixed(2)} MB)`
	);

	// Add metadata
	formData.append(
		"metadata",
		JSON.stringify({
			questions: questions,
			timestamp: new Date().toISOString(),
			totalRecordings: audioResponses.size,
		})
	);

	try {
		console.log("Submitting test...");

		// ✅ Send with timeout and progress tracking
		const response = await axios.post("/submit_test", formData, {
			headers: {
				"Content-Type": "multipart/form-data",
			},
			timeout: 300000, // 5 minute timeout
			onUploadProgress: (progressEvent) => {
				const percentCompleted = Math.round(
					(progressEvent.loaded * 100) / progressEvent.total
				);
				flashMessage.innerHTML = `<div class='spinner'></div> Uploading... ${percentCompleted}%`;
			},
		});

		// Handle success
		if (response.status === 202 || response.status === 200) {
			console.log("Test submitted successfully");
			flashMessage.innerHTML = `<div class="success-message">Success! Redirecting...</div>`;

			// ✅ Wait a moment before redirecting
			setTimeout(() => {
				window.location.href = "/result";
			}, 1000);
		} else {
			throw new Error(`Unexpected status: ${response.status}`);
		}
	} catch (error) {
		console.error("Error submitting test:", error);
		submitButton.disabled = false; // Re-enable button

		if (error.code === "ECONNABORTED") {
			flashMessage.innerHTML = `<div class="error-message">Upload timeout. Please check your connection and try again.</div>`;
		} else if (error.response) {
			flashMessage.innerHTML = `<div class="error-message">Server error: ${error.response.status}</div>`;
		} else if (error.request) {
			flashMessage.innerHTML = `<div class="error-message">Network error. Please check your connection.</div>`;
		} else {
			flashMessage.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
		}
	}
});

// Reset all states when moving to the next question
function resetRecordingState() {
	// Stop any ongoing recording
	if (recorder && recorder.state === "recording") {
		recorder.stop();
	}

	// Stop media stream
	if (mediaStream) {
		mediaStream.getTracks().forEach((track) => track.stop());
		mediaStream = null;
	}

	// Stop any playing audio
	if (audio) {
		audio.pause();
		audio = null;
	}

	// Clear interval
	if (timerInterval) {
		clearInterval(timerInterval);
		timerInterval = null;
	}

	// Reset variables
	recorder = null;
	audioChunks = [];
	audioBlob = null;
	audioURL = null;
	audioSeconds = 0;

	// Reset UI
	stopButton.disabled = true;
	startButton.disabled = false;
	playButton.disabled = true;
	stopAudioButton.disabled = true;
	progressBar.style.width = "0%";
	timerElement.textContent = "0:00";
}

// Initial question update
updateQuestion();

// ✅ Cleanup on page unload
window.addEventListener("beforeunload", (event) => {
	if (audioResponses.size > 0 && audioResponses.size < questions.length) {
		event.preventDefault();
		event.returnValue =
			"You have unsaved recordings. Are you sure you want to leave?";
	}

	// Clean up resources
	resetRecordingState();
});
