from dotenv import load_dotenv
import os
#from transformers import pipeline
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
import speech_recognition as sr
import librosa
import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFaceEndpoint
# Function to transcribe audio using SpeechRecognition
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        text = f"Could not request results from Google Speech Recognition service; {e}"
    return text

# Function to count filler words
def count_filler_words(transcription):
    fillers = ['um', 'uh', 'like', 'you know', 'so', 'actually']
    transcription_lower = transcription.lower()
    words = transcription_lower.split()
    filler_count = sum(words.count(filler) for filler in fillers)
    return filler_count

# Function to analyze audio features
def analyze_audio(file_path, transcription):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate words per minute (WPM)
    words = len(transcription.split())
    pace_wpm = (words / duration) * 60


    # Detect silent periods
    intervals = librosa.effects.split(y, top_db=20)
    silent_time = duration - sum([(end - start) / sr for start, end in intervals])

    # Calculate fluency
    num_pauses = len(intervals) - 1  # Number of pauses is one less than number of intervals
    if num_pauses > 0:
        pauses = [(intervals[i+1][0] - intervals[i][1]) / sr for i in range(num_pauses)]
        average_pause_duration = np.mean(pauses)
        pause_distribution = (min(pauses), max(pauses), average_pause_duration)

        # Determine fluency descriptor
        if average_pause_duration < 0.1:
            fluency_descriptor = "Continuous speech with very short pauses"
        elif average_pause_duration < 0.5:
            fluency_descriptor = "Smooth speech with regular pauses"
        else:
            fluency_descriptor = "Frequent pauses or deliberate speech"

        fluency = (f"{fluency_descriptor}. Average pause duration: {average_pause_duration:.2f} sec, "
                   f"Number of pauses: {num_pauses}, "
                   f"Pause distribution (min, max, mean): ({pause_distribution[0]:.2f}, "
                   f"{pause_distribution[1]:.2f}, {pause_distribution[2]:.2f})")
    else:
        fluency = "Continuous speech or no pauses detected"

    # Count filler words
    filler_count = count_filler_words(transcription)
    filler_analysis = f"Filler words count: {filler_count}"

    # Analyze tone and pitch modulation
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitch[pitch > 0]  # Remove zero values
    if len(pitch_values) > 0:
        avg_pitch = np.mean(pitch_values)
        pitch_range = np.ptp(pitch_values)
        pitch_modulation = np.std(pitch_values)  # Standard deviation for modulation
    else:
        avg_pitch = pitch_range = pitch_modulation = 0
    tone = (f"Average pitch: {avg_pitch:.2f} Hz, Pitch range: {pitch_range:.2f} Hz, "
            f"Pitch modulation (std dev): {pitch_modulation:.2f} Hz")

    # Analyze volume variability
    rms = librosa.feature.rms(y=y)[0]  # Root Mean Square for volume
    volume_variability = np.std(rms)  # Standard deviation for volume changes
    volume_analysis = f"Volume variability (std dev): {volume_variability:.2f}"

    return pace_wpm, fluency, filler_analysis, tone, silent_time, volume_analysis

# Function to generate feedback using Hugging Face model
def generate_feedback(transcription, pace_wpm, fluency, filler_analysis, tone, silent_time, volume_analysis):
    
    load_dotenv()
    huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=huggingface_api_key)
   
    
    # Define the question with all the input parameters
    question = (
        f"Provide a detailed alternative and feedback on the transcribed audio text. "
        f"Assess it based on the given parameters for public speaking: "
        f"Transcription: {transcription}, Pace: {pace_wpm:.2f} words per minute, "
        f"Fluency: {fluency}, Filler Words: {filler_analysis}, Tone: {tone}, "
        f"Silent Time: {silent_time:.2f} sec, Volume Analysis: {volume_analysis}. "
        f"Please respond with a structured paragraph discussing clarity, engagement, and effectiveness."
    )
    
    # Define the template for the prompt
    template = """Question: {question}

    // Answer: Please provide a comprehensive analysis in paragraph form. Consider the various aspects of public speaking and elaborate on how each parameter affects the overall performance."""

    # Create a PromptTemplate instance
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Create an LLMChain with the prompt and language model
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Invoke the LLMChain with the question and get the response
    response = llm_chain.invoke({"question": question})

    # Access the generated text from the response dictionary
    generated_text = response['text']

    # Function to split text into lines with a maximum of 15 words each
    def split_into_lines(text, max_words_per_line=15):
        words = text.split()
        lines = []
        for i in range(0, len(words), max_words_per_line):
            line = " ".join(words[i:i + max_words_per_line])
            lines.append(line)
        return "\n".join(lines)

    # Split the response into multiple lines
    formatted_response = split_into_lines(generated_text)

    return formatted_response



def main():

    st.title("FEEDBACK FOR PUBLIC SPEAKING")

    audio_file = st.file_uploader("Upload an audio file", type=["wav"])

    if audio_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())
        
        transcription = transcribe_audio("temp_audio.wav")
        pace_wpm, fluency, filler_analysis, tone, silent_time, volume_analysis = analyze_audio("temp_audio.wav", transcription)
        
        st.subheader("Transcription")
        st.write(transcription)
        
        st.subheader("Pace")
        st.write(f"{pace_wpm:.2f} words per minute")
        
        st.subheader("Fluency")
        st.write(fluency)
        
        st.subheader("Filler Words")
        st.write(filler_analysis)
        
        st.subheader("Tone")
        st.write(tone)
        
        st.subheader("Silent Time")
        st.write(f"{silent_time:.2f} sec")
        
        st.subheader("Volume Analysis")
        st.write(volume_analysis)
        
        feedback = generate_feedback(transcription, pace_wpm, fluency, filler_analysis, tone, silent_time, volume_analysis)
        
        st.subheader("Feedback")
        st.write(feedback)

if __name__ == "__main__":
    main()



