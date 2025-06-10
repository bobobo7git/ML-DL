from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict
import re
# ✅ 대화 메모리 초기화
chat_memory: Dict[str, list[str]] = {}
print("🚨 [chat_memory] 상담 대화 기록 초기화 완료 – 새로운 세션으로 시작합니다.")
app = FastAPI()

# === CORS 설정 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 모델, 데이터 로딩 ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b-it-qat"

retriever_model = SentenceTransformer("BAAI/bge-m3")
index = faiss.read_index("qa_index_bge_m3.faiss")
with open("qa_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

chat_memory: Dict[str, list[str]] = {}

# === 유사문서 검색 ===
def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[str]:
    query_vec = retriever_model.encode([f"질문: {query}"], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    _, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

# === 프롬프트 생성 ===
def build_rag_prompt(
    query: str,
    retrieved_docs: list[str],
    history_text: str,
    nickname: str,
    user_context: dict,
    emotion: dict
) -> str:
    context = "\n\n".join(retrieved_docs)

    profile_info = []
    if user_context.get("age"):
        profile_info.append(f"{user_context['age']} 연령대")
    if user_context.get("gender"):
        profile_info.append(f"{user_context['gender']} 성별")
    if user_context.get("career"):
        profile_info.append(f"{user_context['career']} 직업")
    if user_context.get("mbti"):
        profile_info.append(f"{user_context['mbti']} 성격유형")
    if user_context.get("stressReason"):
        profile_info.append(f"스트레스 요인: {user_context['stressReason']}")

    profile_summary = ", ".join(profile_info)
    emotion_summary = (
        f"감정 분석 결과: "
        f"우울({emotion.get('sadness', 0)}), 행복({emotion.get('happiness', 0)}), "
        f"분노({emotion.get('angry', 0)}), 중립({emotion.get('neutral', 0)}), 기타({emotion.get('other', 0)})"
    )

    return f"""
너는 따뜻하고 신뢰감 있는 AI 심리상담사야.
말투는 {user_context.get("botCustom", "상냥한")} 톤으로 유지해.

상담자는 "{nickname}"이라는 이름으로 불리고 싶어 해. 반드시 이 이름을 사용해서 말해줘.

상담자의 배경 정보: {profile_summary}
{emotion_summary}

지금까지 상담자와 나눈 대화 기록은 다음과 같아:

{history_text}

[상담자 질문]
{query}

[AI 상담사의 답변 지침]
- 반드시 사용자가 “안녕”, “안녕하세요”, “hi” 등 명시적인 인사를 했을 경우에만 가볍게 인사해.
- **인사 이후에도**, 내담자가 말하지 않은 감정, 배경, 과거 사건 등은 절대 유추하지 마.
- 특히, 인사에 이어 “요즘 힘드셨죠?”, “회사 일이 힘드셨죠?”, “꿈을 잊으셨죠?” 등의 표현은 **금지야.**
- 인사 이후에는 반드시 “무엇이 가장 마음에 걸리셨을까요?”, “어떤 이야기부터 나눠볼까요?”처럼 **중립적인 질문으로 연결해.**

❌ 예시 (잘못된 응답):
질문: 안녕하세요  
응답: 안녕하세요, 내담이님. 요즘 많이 힘드셨죠? 회사도 그렇고...

✅ 예시 (좋은 응답):
질문: 안녕하세요  
응답: 안녕하세요, {nickname}님. 반갑습니다. 어떤 이야기부터 나눠볼까요?

- 사용자가 말하지 않은 내용(회사, 팀원, 집, 꿈, 우울함 등)은 절대 말하지 마.

- 정보가 부족할 경우 추측하지 말고, 상황에 맞는 간단한 질문을 제시해.

- 내담자의 감정 상태는 반드시 **발화에 나타난 직접적 표현**에 기반하여 추측해.  
  예: “너무 힘들다”라고 했다면 → “그만큼 힘든 시간이셨던 것 같아요”와 같이 반영하되, **배경 유추 금지.**

- 사용자가 본인이 했던 질문을 기억하냐고 물어보면, 반드시 사용자의 질문이 무엇이었는지 기억해서 알려줘. 다른 말을 하지마.
  예: "방금 전에 저에게 요즘 너무 힘들다고 말씀해주셨어요."

- 절대 말하지 않은 배경, 장소, 관계(예: 학교, 집, 가족, 친구, 직장 등)를 **추론하거나 언급하지 마.**  
  ❌ “학교와 집안 모두에서 어려움을 겪고 계시고...” ← 이런 문장은 금지

- 내담자가 요청하지 않은 **해결책이나 조언을 제시하지 마.**  
  예: “일기를 써보세요”, “전문가와 이야기해보세요” 등은 말하지 마

- 위로 표현도 **한두 문장 이상 반복하지 마.**  
  예: “혼자가 아니에요”, “곁에 있어요” 같은 문장은 최대 1회 이하로만 표현하고, **중복된 표현은 피해야 함**

- 아래 참고 문서의 답변 스타일(A)을 참고해. 다만, **사용자의 입력이 우선이며, 참고 문서에 있는 감정이나 사건을 끌어오지 마.**

- 반드시 '{nickname}님'이라고 부르고, 응답은 3~5문장, 하나의 문단으로 간결하게 마무리해.  
  문장이 너무 길거나 단락이 여러 개인 답변은 불안감을 줄 수 있으니 지양해.

[❗ 예시]

질문: 나 너무 힘들다!!!  
잘못된 응답 ❌: 학교와 집안에서도 어려움을 겪고 계시고...  
잘된 응답 ✅: 그렇게 말씀하신 걸 보니 정말 지치셨던 것 같아요. 어떤 부분이 가장 힘들게 느껴졌을까요?

→ 반드시 “사용자의 입력 범위 내에서만” 응답을 생성하고, 그 외의 해석은 하지 마.


[❗ 절대 금지 사항 예시]
❌ 사용자가 말하지 않은 환경(예: 집안, 부모) 유추  
❌ 정해진 듯한 조언 ("작은 것부터 시작해보세요" 등 반복 문구)  
✅ 사용자의 표현을 있는 그대로 반영 ("너무 힘들다" → "그렇게 말씀하신 걸 보니 정말 지치셨던 것 같아요")

[참고 문서]
아래는 유사한 상담 문서에서 발췌한 실제 응답 예시야.  
→ 단, 표현 방식만 참고하고, 여기에 있는 내용(장소, 인물, 사건 등)은 절대 가져오지 마.

{context}
"""

# === /ai-data/chat 엔드포인트 ===


@app.post("/chat")
async def chat(request: Request):
    
    body = await request.json()
    nickname = body.get("nickname", "상담자님")
    message = body.get("message", "")
    user_context = body.get("userContext", {}) or {}
    emotion = body.get("emotion", {}) or {}
    print("현재 대화 이력:", chat_memory.get(nickname))
    if nickname not in chat_memory:
        chat_memory[nickname] = []


    # 대화 기록 업데이트
    chat_memory[nickname].append(f"{nickname}: {message}")
    history_text = "\n".join(chat_memory[nickname][-6:])
    retrieved_docs = retrieve_relevant_chunks(message)

    # 프롬프트 생성
    prompt = build_rag_prompt(
        query=message,
        retrieved_docs=retrieved_docs,
        history_text=history_text,
        nickname=nickname,
        user_context=user_context,
        emotion=emotion
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "num_predict": 250,
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.7
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(OLLAMA_URL, json=payload)
            result = res.json()
            ai_text = result.get("response", "").strip()
    except Exception as e:
        print("[❌ chat 예외 발생]", str(e))
        return {"ai_response": "모델 응답 중 오류가 발생했습니다."}

    chat_memory[nickname].append(f"AI: {ai_text}")
    return {"ai_response": ai_text}


# 📋 요약 프롬프트
def build_summary_prompt(history_text: str) -> str:
    return f"""
너는 지금까지의 상담 대화를 바탕으로 상담자님의 감정 상태를 분석하는 심리상담사야.
Russell 감정 원형 모형(Circumplex Model of Affect)에 따라 감정을 분석해.

아래는 상담자와 AI의 전체 대화 내용이야:

{history_text}

이 상담 내용을 기반으로, 아래 JSON 형식으로 요약 보고서를 작성해줘.
이 상담 내용을 기반으로 아래 형식에 맞는 JSON 리포트를 작성해줘. 각 필드는 다음을 충실히 반영해야 해:

---

📌 "summary": 전체 상담 내용을 객관적으로 요약해. 감정 분석은 하지 말고, 어떤 주제로 어떤 이야기들이 오갔는지만 간결하게 정리해.

📌 "analyze": 대화 중 상담자의 감정이 어떻게 흘러갔는지 분석해줘. 예를 들어, 처음에는 불안했지만 점차 안정을 찾았다거나, 분노가 점점 줄어들고 무기력함이 드러났다는 식으로 시간 순서의 감정 변화를 서술해.

📌 "valence": Russell 감정 원형 모델 기준으로 상담자의 전반적인 감정 방향을 평가해. (예: "positive", "neutral", "negative")

📌 "arousal": Russell 감정 원형 모델 기준으로 감정의 활성도를 평가해. (예: "high", "medium", "low")

---

✅ 반드시 아래 JSON 형식으로만 출력하고, 설명 문장이나 여분의 텍스트는 절대 포함하지 마.  
✅ JSON 키 이름은 반드시 영문 소문자로 유지하고, 순서도 유지해.

```json
{{
  "summary": "대화 요약 (예: 상담자는 최근 업무 스트레스로 인해 불면을 겪고 있음을 이야기했다...)",
  "analyze": "감정 변화 분석 (예: 초반에는 분노가 있었지만 점점 우울과 무기력함이 강조되었다...)",
  "valence": "positive | neutral | negative",
  "arousal": "high | medium | low"
}}
```
"""

@app.post("/summary")
async def summarize(request: Request):
    body = await request.json()
    nickname = body.get("nickname", "default")

    if nickname not in chat_memory or not chat_memory[nickname]:
        return {"summary": "요약할 상담 기록이 없습니다."}

    history_text = "\n".join(chat_memory[nickname])
    prompt = build_summary_prompt(history_text)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "num_predict": 400,
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.7
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(OLLAMA_URL, json=payload)
            result = res.json()
            response_text = result.get("response", "").strip()

            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                match = re.search(r"\{[\s\S]*\}", response_text)
                if match:
                    return json.loads(match.group())
                else:
                    return {
                        "summary": "JSON 형식을 파싱할 수 없습니다.",
                        "raw_response": response_text
                    }

    except Exception as e:
        print("[❌ 요약 오류 발생]", str(e))
        return {
            "summary": "레포트를 생성하는 중 오류가 발생했습니다.",
            "raw_response": response_text if "response_text" in locals() else "없음"
        }

