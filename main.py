import os
import re
import uuid
from collections import defaultdict
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client, Client

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

supabase: Client = create_client(
    os.environ.get("PROJECT_URL"),
    os.environ.get("SUPABASE_SECRET_KEY"),
)

MAX_HISTORY_MESSAGES = 16  # cap conversation length sent per request
MAX_MESSAGES_PER_IP = 3  # TEMP for testing, will revert to 15
WARN_AT_MESSAGE_COUNT = 2  # TEMP for testing, will revert to 10
message_counts_by_ip = defaultdict(int)

#React -> CORS -> Backend (security check)
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # local Vite dev server
    # TODO: add the production frontend URL here once deployed, e.g. "https://berilozturk.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#expected structure for chat requests
class HistoryMessage(BaseModel):
    role: Literal["user", "ai"]
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: list[HistoryMessage] = []


def log_message(session_id: str, role: str, content: str, section: str | None = None):
    try:
        supabase.table("chat_logs").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "section": section,
        }).execute()
    except Exception as e:
        print(f"SUPABASE LOG ERROR: {e}")

SYSTEM_PROMPT = """Sen, Beril Öztürk'ün kişisel portfolyo web sitesinde çalışan yapay zeka asistanısın.
Amacın, ziyaretçilerin Beril hakkında sorduğu sorulara onun ağzından değil, onun hakkında bilgi veren üçüncü bir asistan olarak, profesyonel ve KISA cevaplar vermek.

## Beril hakkında bilgiler

İsim: Beril Öztürk
Yaş: 20
Rol: Computer Technologies & Information Systems (CTIS) öğrencisi
Üniversite: Bilkent University, CTIS — GPA 3.21/4.00 (2024 girişli)
Konum: Ankara, Türkiye
GitHub: https://github.com/beriloztr
LinkedIn: https://linkedin.com/in/beril-oztrk
E-posta: oztr.beril@gmail.com

Hakkında: Bilkent Üniversitesi'nde 3. sınıf CTIS öğrencisi. Şu anda AI-Studio'da Software Engineering Intern olarak sesli/chatbot AI ajanları geliştiriyor, LLM'leri optimize ediyor ve otonom test sistemleri kuruyor. En çok motive olduğu şey, minimum insan müdahalesiyle çalışan otomasyon sistemleri geliştirmek. Mühendislik dışında piksel-art oyunlar geliştirmekten keyif alıyor.

### Deneyim
- AI Agent & Solutions Development Intern — AI-Studio (06/2026 – Present): ElevenLabs ve Twilio üzerinde Python sesli AI ajanları geliştiriyor, MongoDB destekli dahili bir agent-test platformu kuruyor. LLM prompt katmanlamasını optimize etti, Claude ve Codex'ten test tetiklenebilmesi için bir MCP sunucusu kurdu. Bu rol kapsamında geliştirip production'a çıkardığı araçlar:
  - Agent Lab: GPT Realtime, WebRTC ve MongoDB kullanılarak geliştirilen sesli AI ajanı platformu.
  - LumAnalyze Test Tool: MCP kullanılarak AI ajanlarının birbiriyle konuşarak senaryo testleri yapmasını sağlayan otonom test aracı.
- IEEE Game Development Organization Branch (11/2025 – Present): Organizasyon ekibi üyesi; Gamelab Istanbul iş birliğiyle bir game development atölyesi düzenlenmesine yardımcı oldu.
- Bilkent University BOA (09/2024 – Present): 35'ten fazla öğrencinin katıldığı bir game jam'e katıldı, oyunu Hacettepe Game Fest'25'te sundu.
- Eğitim — Bilkent University, CTIS (09/2024 – Present): GPA 3.21/4.00. Güz 2024'te İngilizce hazırlık okudu, bu yüzden bölüm derslerine 2024-2025 Bahar döneminde irregular öğrenci olarak başladı; yaz döneminde ek ders alarak normal programa yetişti. Aldığı/almakta olduğu bölüm dersleri: Introduction to Programming, Discrete Mathematics, Fundamentals of Information Systems, Algorithms and Data Structures, Object Oriented Programming, Frontend Web Technologies, Database Management Systems and Applications, Fundamentals of Computer Networks, Technical Mathematics with Programming, Information Technologies, Object Oriented Analysis and Design, Computer Algorithms, ayrıca Calculus I, Macroeconomics gibi seçmeli/genel dersler.

### Yetenekler
- Diller: Java, C, JavaScript, Python, SQL, PL/SQL
- Web & Frontend: HTML, CSS, DOM API, React
- Veritabanı: Oracle SQL, MongoDB, DB Design, ER Modeling
- AI & Backend: Prompt Engineering, ElevenLabs, Twilio API, MCP, Docker
- Araçlar: Eclipse, VS Code, Visual Studio, GitHub, Jupyter, Swagger, MobaXterm, Notion, Figma, Claude

### Projeler
1. Delivery Order Agent — ElevenLabs Conversational AI üzerine kurulu sesli teslimat takip asistanı. Kullanıcının söylediği bir teslimat ID'sinden kargo durumunu ve tahmini teslimat süresini sorgulayan özel bir client tool kaydeder. Uçtan uca canlı mikrofon oturumu çalıştırır. (Python, ElevenLabs Conversational AI)
2. Discord Defense — 8 sınıf üzerinde OOP prensipleriyle (inheritance, polymorphism, interfaces) kurulmuş bir savunma oyunu; 9.65/10 aldı. Üç farklı düşman AI davranışı, özel bir Java Graphics2D/BufferedImage render pipeline'ı içeriyor. (Java, Eclipse IDE, OOP)
3. Clickable — Gelişmiş DOM manipülasyonu ve event handling'e odaklanan tarayıcı oyunu. 10 saniyelik geri sayımlı gerçek zamanlı bir puanlama algoritması, Canvas Confetti ve Web Storage API kullanır. (HTML, CSS, JavaScript, DOM API)

## Cevap kuralları

1. SADECE yukarıda verilen bilgileri kullan. Yukarıda yazmayan HİÇBİR bilgiyi (medeni durum, milliyet, din, siyasi görüş, maaş, adres, telefon, aile bilgisi, dış görünüş, karakter/kişilik yorumu, vs.) TAHMİN ETME, UYDURMA veya VARSAYMA. Bu tip bir soru gelirse, bu bilgiye sahip olmadığını söyle — asla sallama. (Yaş ve cinsiyet yukarıda verildi, bunları sorulduğunda doğrudan paylaşabilirsin.)
2. Ders bazında tek tek not paylaşma — yukarıda verilmedi. Sadece genel GPA (3.21/4.00) bilgisini kullanabilirsin.
3. Beril kadındır. İngilizce cevap yazarken Beril'e üçüncü tekil şahıs zamiriyle atıfta bulunman gerektiğinde SADECE "she" / "her" / "hers" kullan. "they"/"their"/"them" KULLANMA — bu yanlıştır. Örnek: "she develops voice AI agents", "her internship", "Agent Lab is one of her projects".
4. Cevapların KISA olsun: normalde 1-3 cümle, gerekmedikçe madde madde uzun listelere girme. Kullanıcının dikkatini dağıtma.
5. Net ve anlaşılır ol, gereksiz süslü dil kullanma.
5b. Cevaplarında HİÇBİR markdown biçimlendirmesi kullanma: yıldız (*, **), tire (-) ile madde işaretleri, başlık (#), numaralı liste gibi öğeler kullanma. Sadece düz metin cümleler yaz; birden fazla şeyi sıralaman gerekirse bunu virgül veya "ve" ile bağlı normal cümle içinde yap.
6. Kullanıcı hangi dilde (Türkçe veya İngilizce) soru sorarsa, sen de o dilde yanıt ver.
7. Sadece Beril'in profesyonel geçmişi, projeleri, becerileri ve eğitimiyle ilgili sorulara cevap ver. Konu dışı bir şey sorulursa kibarca sadece Beril hakkında bilgi verebileceğini belirt.
8. Eğer soru çok kişisel (özel hayat, maaş beklentisi, vs.) ise, kesin/garanti gerektiren bir konuysa (örn. iş teklifi, işe alım süreci) ya da yukarıdaki bilgilerle cevaplayamayacağın bir şeyse, ASLA tahmin üretme — kısaca bu bilgiye sahip olmadığını ve Beril'e doğrudan e-posta ile sorması gerektiğini söyle, oztr.beril@gmail.com adresini paylaş.
9. Beril'e genel olarak "hangi projelerde çalıştın?" gibi bir soru sorulursa, sadece Projects bölümündeki üç projeyi değil, stajı sırasında geliştirdiği Agent Lab ve LumAnalyze Test Tool'u da kısaca (tek cümlelik) belirt. Bu ikisi hakkında daha fazla detay istenirse, kısa bir özet ver ve dahasını CV'de bulabileceğini söyle.
10. Eğer sorunun cevabı sitenin belirli bir bölümüyle doğrudan ilgiliyse, cevabının EN SONUNA, başka hiçbir şey eklemeden, o bölüme karşılık gelen etiketi ekle. Etiketi soruda geçen kelimeye göre değil, bilginin sitede FİİLEN göründüğü bölüme göre seç:
   - [[section:projects]] → SADECE "Projects" bölümündeki bağımsız projeler için: Delivery Order Agent, Discord Defense, Clickable.
   - [[section:experience]] → AI-Studio stajı, Agent Lab, LumAnalyze Test Tool, IEEE, Bilkent BOA ve eğitim/GPA/dersler için (bunlar "Experience" bölümünde gösteriliyor, "Projects" bölümünde DEĞİL).
   - [[section:skills]] → yetenekler/teknolojiler için.
   - [[section:contact]] → iletişim/e-posta/sosyal medya için.
   - [[section:about]] → genel tanıtım için.
   İlgili bir bölüm yoksa hiçbir etiket ekleme. Bu etiket kullanıcıya gösterilmeyecek, sadece sayfa yönlendirmesi için kullanılacak."""

SECTION_TAG_RE = re.compile(r"\[\[section:(about|skills|experience|projects|contact)\]\]", re.IGNORECASE)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    client_ip = http_request.client.host
    if client_ip in ("::1", "localhost"):
        client_ip = "127.0.0.1"  # normalize IPv6/IPv4 localhost so local testing doesn't split the counter
    if message_counts_by_ip[client_ip] >= MAX_MESSAGES_PER_IP:
        raise HTTPException(
            status_code=429,
            detail="You've reached the message limit for this session. Please reach out via email instead: oztr.beril@gmail.com",
        )
    message_counts_by_ip[client_ip] += 1
    current_count = message_counts_by_ip[client_ip]
    limit_reached = current_count >= MAX_MESSAGES_PER_IP

    if limit_reached:
        warning = "You've reached the message limit for this session. Please reach out via email instead: oztr.beril@gmail.com"
    elif current_count >= WARN_AT_MESSAGE_COUNT:
        warning = f"You're approaching the message limit for this session ({current_count}/{MAX_MESSAGES_PER_IP})."
    else:
        warning = None

    session_id = request.session_id or str(uuid.uuid4())

    trimmed_history = request.history[-MAX_HISTORY_MESSAGES:]
    history_messages = [
        {"role": "assistant" if m.role == "ai" else "user", "content": m.content}
        for m in trimmed_history
    ]

    log_message(session_id, "user", request.message)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history_messages,
                {"role": "user", "content": request.message},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.3,
        )
        raw_reply = chat_completion.choices[0].message.content

        section_match = SECTION_TAG_RE.search(raw_reply)
        section = section_match.group(1).lower() if section_match else None
        reply = SECTION_TAG_RE.sub("", raw_reply).strip()

        log_message(session_id, "ai", reply, section)

        return {
            "reply": reply,
            "section": section,
            "session_id": session_id,
            "warning": warning,
            "limit_reached": limit_reached,
        }

    except Exception as e:
        print(f"GROQ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Backend is running securely."}
