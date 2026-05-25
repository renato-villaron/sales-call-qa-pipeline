 Sales Call QA Pipeline — Automated Quality Analysis at Scale

Replaced a 10-person monitoring team by automating the end-to-end analysis of 21,000+ sales calls over 7 months using Whisper + GPT and Python.


 The Problem
The company processed ~140 sales calls per day, Monday through Friday. Manually reviewing each call for quality assurance required a large team listening to hours of audio — slow, expensive, and unscalable.
Key questions that needed answering at scale:

Did the sales rep follow the approved script?
Was the sale conducted ethically (no fraudulent inducement)?
Did the contract information match what was verbally agreed to on the call?


⚙️ Solution Overview
A fully automated Python pipeline that:

Ingests audio files from Google Drive
Transcribes each call using OpenAI Whisper (running on Google Colab T4 GPU)
Analyzes each transcript with GPT-4o mini using custom QA prompts
Exports all results into a single structured Excel file for analysis

Google Drive (audios)
        ↓
  Google Colab (T4)
        ↓
  Whisper STT → transcript.txt (per call)
        ↓
  GPT-4o mini → structured QA analysis
        ↓
  ETL script → consolidated Excel report
        ↓
  Power BI / manual review

 Tech Stack
LayerToolSpeech-to-TextOpenAI Whisper (large-v2)LLM AnalysisGPT-4o mini via OpenAI APIRuntimeGoogle Colab (T4 GPU)StorageGoogle DriveETL & ExportPython (pandas, openpyxl)Final OutputExcel (.xlsx)

 QA Analysis Criteria
Each call is evaluated by GPT on the following dimensions:

Script Adherence — Did the rep follow the approved sales script?
Fraud Detection — Were any prohibited sales tactics or false inducements used?
Data Accuracy — Does the verbal agreement match the contract details (name, plan, price)?
Communication Quality — Clarity, professionalism, and customer handling

Each output is structured so results can be aggregated across hundreds of calls in a single Excel sheet.

 Results
MetricValueCalls processed~21,000+Daily volume~140 calls/dayDuration7 months in productionTeam replaced~10 QA monitors (CLT, 6×1 schedule)Time to analyze 140 callsMinutes (vs. a full workday manually)

 How to Run
Prerequisites
bashpip install openai openai-whisper pandas openpyxl google-auth google-api-python-client
1. Transcription
pythonimport whisper

model = whisper.load_model("large-v2")
result = model.transcribe("call_audio.mp3")

with open("call_transcript.txt", "w") as f:
    f.write(result["text"])
2. QA Analysis with GPT
pythonfrom openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

with open("call_transcript.txt") as f:
    transcript = f.read()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a sales quality analyst. Analyze the transcript and return a structured JSON with: script_adherence (yes/no), fraud_detected (yes/no), data_accuracy (yes/no), notes."
        },
        {
            "role": "user",
            "content": transcript        }
    ]
)

print(response.choices[0].message.content)
3. Consolidate to Excel
pythonimport pandas as pd
import json

results = []  # list of GPT JSON responses per call

for call_result in results:
    parsed = json.loads(call_result)
    results.append(parsed)

df = pd.DataFrame(results)
df.to_excel("qa_report.xlsx", index=False)

 Key Takeaways

No dedicated GPU required — Google Colab T4 handles Whisper at scale for free
Prompt engineering matters — structured JSON outputs from GPT make ETL trivial
LLMs + audio = powerful QA — this approach generalizes to any industry with high call volume (insurance, banking, telecom, etc.)


 Contact
Built by [Renato Villaron] · [joserenatovillaron@gmail.com]
Open to remote opportunities in Revenue Operations, Sales Analytics, and Data Engineering.
