from flask import Flask, render_template, request, jsonify
import csv
import os
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv  # <- import dotenv

# Load environment variables from .env
load_dotenv()  # <- this reads .env file

app = Flask(__name__)

# Read the API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Make sure you have a .env file with the key.")

# Initialize Groq client
client = Groq(api_key=api_key)

def analyze_transcript(transcript):
    """Analyze transcript using Groq API"""
    try:
        prompt = f"""
        Analyze this customer service call transcript and provide:
        1. A 2-3 sentence summary of the conversation
        2. The customer's overall sentiment (positive, neutral, or negative)
        
        Transcript: {transcript}
        
        Please format your response as:
        SUMMARY: [summary here]
        SENTIMENT: [sentiment here]
        """
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=500,
            top_p=1,
            stream=False,
        )
        
        response = chat_completion.choices[0].message.content
        summary, sentiment = "", ""
        
        for line in response.split('\n'):
            if line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('SENTIMENT:'):
                sentiment = line.replace('SENTIMENT:', '').strip()
        
        return summary, sentiment
        
    except Exception as e:
        return f"Error analyzing transcript: {str(e)}", "Error"

def save_to_csv(transcript, summary, sentiment):
    filename = "call_analysis.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Timestamp', 'Transcript', 'Summary', 'Sentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Transcript': transcript,
            'Summary': summary,
            'Sentiment': sentiment
        })
    
    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    transcript = request.form.get('transcript', '')
    if not transcript:
        return jsonify({'error': 'No transcript provided'})
    
    summary, sentiment = analyze_transcript(transcript)
    filename = save_to_csv(transcript, summary, sentiment)
    
    return jsonify({
        'transcript': transcript,
        'summary': summary,
        'sentiment': sentiment,
        'filename': filename
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
