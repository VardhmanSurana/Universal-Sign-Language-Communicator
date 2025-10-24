***

# Universal Sign Language Communicator

This project is a **bidirectional translation system** designed to bridge the communication gap between spoken language and Indian Sign Language (ISL). It provides a seamless, real-time solution for two-way communication, enabling spoken language users to understand ISL and ISL users to understand spoken language.

The system integrates speech recognition, Natural Language Processing (NLP), and deep learning-based gesture recognition to function as a complete communication aid.

## Features

* **Bidirectional Translation:** Provides both **Speech-to-Sign** and **Sign-to-Speech** functionality.
* **Speech-to-Sign Mode:** Converts spoken English into animated Indian Sign Language (ISL) gestures (GIFs).
* **Sign-to-Speech Mode:** Uses a webcam to recognize live ISL hand gestures and converts them into text and natural-sounding spoken audio.
* **Hybrid Online/Offline Support:**
    * **Online:** Utilizes Google Speech API for high-accuracy speech recognition.
    * **Offline:** Employs CMU Sphinx for on-device, English-only speech recognition when no internet is available.
* **Real-time Processing:** Optimized for high-speed, low-latency performance using **MediaPipe** for hand tracking and **TensorFlow Lite** for efficient gesture classification.
* **Intelligent Error Handling:** If a spoken phrase doesn't have a direct sign in the database, the system defaults to spelling the words out using the alphabetic ISL representation.

---

## How It Works

The system operates in two distinct modes, supported by a user-friendly Tkinter interface. The overall data flow is illustrated below:

 ## Workflow Diagram
```mermaid
graph TD
    User(User)
    %% Define Subgraphs
    subgraph Backend
        direction TD
        %% Path 1: Speech-to-Sign
        Speak[Speak words or phrase]
        SpeechRec[Speech Recognition <br> (Google Speech API / CMU Sphinx)]
        LangDetect[Language Detection <br> (Google Translator API)]
        Translate[Translation <br> (Transformer-based Model)]
        GenerateGIF[Generate GIF from Dataset]
        SpeakAgain{Speak Again?}
        %% Path 2: Sign-to-Speech
        Webcam[Start Webcam]
        Keypoints[Hand keypoint Input <br> (MediaPipe Hand Tracking)]
        Features[Feature Extraction <br> (Keypoint Angles & Distances)]
        Detect[Detect hand gesture using model]
        GenerateSpeech[Generate Speech <br> (Google TTS)]
        StartAgain{Start Again?}
    end

    subgraph Storage
        Dataset[GIF Dataset]
    end

    %% Define End State
    ExitApp[Exit]
    End((End))

    %% Define Connections
    User -- Selects Speech-to-Sign --> Speak
    Speak --> SpeechRec
    SpeechRec --> LangDetect
    LangDetect --> Translate
    Translate --> GenerateGIF
    Dataset -- Provides GIFs --> GenerateGIF
    GenerateGIF -- Displays GIF to --> User
    GenerateGIF --> SpeakAgain
    SpeakAgain -- Yes --> Speak
    SpeakAgain -- No --> ExitApp

    User -- Selects Sign-to-Speech --> Webcam
    Webcam -- Captures User's Gesture --> Keypoints
    Keypoints --> Features
    Features --> Detect
    Detect --> GenerateSpeech
    GenerateSpeech -- Plays Audio to --> User
    GenerateSpeech --> StartAgain
    StartAgain -- Yes --> Webcam
    StartAgain -- No --> ExitApp

    ExitApp --> End
```
### 1. Speech-to-Sign Module

This mode converts spoken words into visual ISL gestures.

1.  **Speech Input:** The user speaks a phrase. The audio is captured using the microphone.
2.  **Speech Recognition:** The system uses the **Google Speech API** (online) or **CMU Sphinx** (offline) to transcribe the audio into text.
3.  **NLP Processing:** The transcribed text undergoes NLP processing to make it compatible with ISL grammar. This includes removing stop words and adjusting tenses.
4.  **Gesture Generation:** The system searches its database for the corresponding ISL animation (GIF) for the processed text and displays it on the screen.

### 2. Sign-to-Speech Module

This mode converts ISL gestures from a webcam into spoken audio.

1.  **Video Capture:** The system activates the webcam to monitor the user's hand movements.
2.  **Hand Tracking:** **MediaPipe** is used to detect and track 21 key hand landmarks in real-time from the video feed.
3.  **Gesture Classification:** The extracted landmark data is fed into a **TensorFlow Lite-based CNN model**, which classifies the hand gesture.
4.  **Speech Output:** The classified gesture is mapped to its corresponding English word or phrase. This text is then converted into natural-sounding speech using the **Google Text-to-Speech (TTS) API**.

---

## Dataset

The dataset used for training the Sign-to-Speech model can be found here: [WLASL Processed Dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

## How to Add New Videos to the Dataset

To expand the dataset with new signs, follow these steps:

1.  Place your `.mp4` video files into the `Sign-to-speech/data/videos` directory.
2.  Ensure the video file names are descriptive (e.g., `hello-01.mp4`, `goodbye-01.mp4`). The system will use the part before the first hyphen as the sign name.
3.  The system will automatically process these videos and extract hand landmarks when the `load_dataset()` function is executed (e.g., during model training or reference sign creation).

## How to Run the Program

To run the Universal Sign Language Communicator, execute the `main.py` file located in the root directory of the project. This will launch the graphical user interface (GUI) from which you can select either "Speech to Sign" or "Sign to Speech" mode.

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r Requirements.txt

# Run the application
python main.py
```

---

## Technology Stack
* **Core Language:** Python
* **Gesture Recognition:** MediaPipe 
* **Deep Learning Model:** TensorFlow Lite (TFLite) 
* **Speech-to-Text (Online):** Google Speech API 
* **Speech-to-Text (Offline):** CMU Sphinx 
* **Text-to-Speech:** Google Text-to-Speech (gTTS) API 
* **User Interface (GUI):** Tkinter 
* **NLP:** Custom scripts for ISL grammatical alignment.

---

## Performance

The system was evaluated for accuracy, speed, and efficiency.

* **Sign-to-Speech Accuracy:** The gesture recognition model achieved **96.4% classification accuracy** on the ISL dataset.
* **Speech-to-Sign Accuracy:** The module achieved a **90.8% translation accuracy**, with a low Word Error Rate (WER) of 7.9% and a high BLEU score of 0.81.
* **Real-time Latency:**
    * Sign-to-Speech: ~**0.75 seconds** per gesture.

    * Speech-to-Sign: ~**1.1 seconds** per phrase.

---

## Future Work

Future plans to enhance this system include:

* **Expanding the Database:** Adding a wider range of ISL phrases, sentences, and expressions.
* **Improved Offline Mode:** Training on-device models for speech recognition and translation to reduce internet dependency.
* **Enhanced Robustness:** Improving speech recognition to handle diverse accents and noisy environments.
* **3D Avatars:** Integrating 3D avatar-based sign synthesis for more expressive and fluid ISL translations.

---

## Authors

This project was developed by:

* **Adarsh Arora** 
* **Ravish Arora** 
* **Vardhman Surana** 
* **Praveen Joe IR** 

*School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India*