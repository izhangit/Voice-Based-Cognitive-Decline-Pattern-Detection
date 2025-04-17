# Voice-Based Cognitive Decline Detection Using NLP and Audio Feature Extraction

## Project Title
Voice-Based Cognitive Decline Detection Using NLP and Audio Feature Extraction

## 🎯 Objective
To build a basic proof-of-concept pipeline using raw voice data samples that detects indicators of cognitive stress or early mental decline. The system combines audio signal analysis and Natural Language Processing (NLP) to identify unusual speech patterns.

## 📌 Problem Statement
You are part of the MemoTag speech intelligence team. Your responsibility is to analyze 5–10 anonymized or simulated voice samples and detect features that might signal cognitive impairment. This includes hesitation markers, speech delays, irregular pitch, missing words, and slower response times.

## 🔁 Workflow Overview

### 1. Preprocess Audio and Transcribe to Text
- **Format:** .wav voice samples
- **Library:** librosa for audio signal loading
- **Library:** SpeechRecognition for converting speech to text using Google's API

### 2. Feature Extraction
#### 🎤 Audio Features
- **Duration:** Length of the audio sample
- **Pause Count:** Number of silence segments (using energy threshold)
- **Average Pitch:** Mean pitch over time
- **Pitch Variability:** Changes in pitch (monotony vs. fluctuation)

#### 💬 Text Features
- **Word Count:** Total spoken words
- **Hesitations:** Count of "uh", "um", etc.
- **Average Word Length:** Complexity of speech
- *(Optional future features: sentence completion errors, word substitutions, etc.)*

## 🤖 Unsupervised Machine Learning
### ✅ Clustering with KMeans
- Grouped the voice samples based on extracted features.
- Used KMeans to form two clusters: one for normal and one for at-risk behavior.
- Applied StandardScaler for normalization.
- Applied PCA (Principal Component Analysis) to reduce feature dimensions and visualize.

## 📉 Visualization
- Used Seaborn and Matplotlib to generate a 2D scatter plot.
- The plot helps in observing natural groupings of speech patterns.

## 🔍 Most Insightful Features

| Feature        | Importance                                         |
|----------------|---------------------------------------------------|
| pause_count    | Higher pauses are potential red flags            |
| hesitations    | Shows hesitation in thinking/speaking             |
| pitch_var      | Low variance may show emotional flatness          |
| word_count     | Very low counts may imply memory issues           |


## 📈 Sample Visualization
- **PCA Scatter Plot:** Shows voice samples clustered as “Normal” and “At Risk”.
- **File saved as:** `cluster_visualization.png`

## 📋 Evaluation Criteria Checklist

| Criteria                     | Covered? ✅ |
|------------------------------|------------|
| Creative feature engineering  | ✅ Yes     |
| Interpretable modeling        | ✅ Yes     |
| Domain relevance              | ✅ Yes     |
| Clean code and documentation  | ✅ Yes     |
| Clear final outputs and visuals| ✅ Yes    |

## 🚀 Future Work & Suggestions
- Use real patient data from hospitals.
- Include task-based prompts like memory recall.
- Leverage advanced NLP (BERT, GPT).
- Track cognitive decline over time.
- Collaborate with neurologists for more speech indicators.

## 📦 Deliverables

| File Name                     | Description                                      |
|-------------------------------|--------------------------------------------------|
| `cognitive_decline_detection.ipynb` | Full pipeline notebook                       |
| `bulk_voice_features.csv`      | Extracted audio and NLP features                |
| `cluster_visualization.png`    | PCA plot of clustered voice samples             |
| `README.md`                   | GitHub documentation                           |
| `REPORT.md / REPORT.pdf`      | Submission-ready report                         |

## 🙋‍♂️ Author
**Izhan Abdullah**  
B.Tech CSE – SRM University  
MemoTag AI/ML Minor Project Submission  
April 2025

