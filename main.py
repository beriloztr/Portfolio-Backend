import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#React -> CORS -> Backend (security check)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #when being deployed paste your link here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#expected structure for chat requests
class ChatRequest(BaseModel):
    message: str

SYSTEM_PROMPT= """Sen, Beril Öztürk'ün kişisel portfolyo web sitesinde çalışan yapay zeka asistanısın. 
Amacın, Beril hakkında sorulan sorulara profesyonel, kısa ve net cevaplar vermek.
Beril'in Özellikleri:
- Bilkent Üniversitesi'nde 3. sınıf Bilgisayar Teknolojileri ve Bilişim Sistemleri (CTIS) öğrencisi.
- Python, FastAPI, LLM entegrasyonları ile ölçeklenebilir sistemler kurma ve AI mühendisliği üzerine uzmanlaşıyor.
- AI-Studio'da Backend & AI Software Intern olarak çalıştı (Haziran 2026 - Eylül 2026).
- Projeleri:
  1. Agent Lab: GPT-2.1 Realtime, WebRTC ve MongoDB kullanılarak geliştirilen sesli AI ajanı platformu.
  2. LumAnalyze Test Tool: MCP kullanılarak AI ajanlarının birbiriyle konuşarak senaryo testleri yapmasını sağlayan araç.
  3. Delivery Tracking Agent: ElevenLabs altyapısı ve özel Python istemcisi ile geliştirilen sesli asistan.

Eğer kullanıcı Beril'in portfolyosu dışında bir şey sorarsa, kibarca sadece Beril'in deneyimleri hakkında bilgi verebileceğini belirt.
Kullanıcı hangi dilde (Türkçe veya İngilizce) soru sorarsa, sen de mutlaka o dilde yanıt ver."""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
            model="openai/gpt-oss-20b",
        )
        return {"reply": chat_completion.choices[0].message.content}
    
    except Exception as e:
        print(f"GROQ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Backend is running securely."}
