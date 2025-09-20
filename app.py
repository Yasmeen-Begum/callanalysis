import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use .env key if available, else fallback to hardcoded key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq API endpoint and model
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"  # or "llama3-70b-8192" for more power


def analyze_transcript(transcript):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    You are an AI assistant. Analyze the following customer call transcript:
    Transcript: \"{transcript}\"

    1. Summarize the conversation in 2–3 sentences.
    2. Extract the customer's sentiment (positive / neutral / negative).
    Respond in the format:
    Summary: <summary>
    Sentiment: <sentiment>
    """

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        result = response.json()

        if "choices" not in result:
            error_msg = result.get("error", {}).get("message", "Unknown error from Groq API.")
            return None, None, error_msg

        content = result['choices'][0]['message']['content']
        summary = content.split("Summary:")[1].split("Sentiment:")[0].strip()
        sentiment = content.split("Sentiment:")[1].strip()

        return summary, sentiment, None

    except Exception as e:
        return None, None, str(e)

# Streamlit UI
st.title("📞 Call Transcript Analyzer")

transcript_input = st.text_area("Enter customer call transcript:")

if st.button("Analyze"):
    if transcript_input:
        summary, sentiment, error = analyze_transcript(transcript_input)

        if error:
            st.error(f"API Error: {error}")
        else:
            st.subheader("📋 Results")
            st.write("**Transcript:**", transcript_input)
            st.write("**Summary:**", summary)
            st.write("**Sentiment:**", sentiment)

            # Save to CSV
            df = pd.DataFrame([[transcript_input, summary, sentiment]],
                              columns=["Transcript", "Summary", "Sentiment"])
            df.to_csv("call_analysis.csv", mode='a', header=not os.path.exists("call_analysis.csv"), index=False)
            st.success("Saved to call_analysis.csv")
    else:
        st.warning("Please enter a transcript.")

