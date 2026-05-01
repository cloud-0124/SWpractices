# ./app/main.py
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.spam import check_spam
from pydantic import BaseModel

import logging
import traceback                      
from app.issue import *

# 1) 로그 포맷: 시간 + 레벨 + 메시지
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | "
           "%(filename)s:%(lineno)d (%(funcName)s) | "
           "%(message)s"
)
logger = logging.getLogger("spamcheck")

# FastAPI 앱 생성
app = FastAPI(title="SpamCheck Web")

# 정적 파일 연결
app.mount("/static", StaticFiles(directory="static"), name="static")

# 메인 페이지
@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

# 요청 모델
class ClassifyRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify(payload: ClassifyRequest):
    text = payload.text

    # (A) 요청 들어온 것 자체를 기록: 언제(로그 시간) / 무엇(endpoint) / 어떤 입력
    logger.info(f"CALL /classify | text='{text}' | len={len(text)}")

    try:
        if text == "crash":
            raise RuntimeError("의도적 장애 추가")

        label, score = check_spam(text)

        # (B) 정상 처리 결과도 짧게 기록
        logger.info(f"OK /classify | label={label} score={score}")

    except Exception as e:
        # (C) 디버깅 핵심: 에러 종류/메시지 + 스택트레이스(파일/라인 포함)
        # logger.exception은 현재 예외의 traceback을 자동으로 찍어줍니다.
        logger.exception(
            f"FAIL /classify | text='{text}' | error={type(e).__name__}: {e}"
        )

        # (D) GitHub Issue 자동 생성
        tb = traceback.format_exc()
        title = f"[Prod Error] /classify failed: {type(e).__name__}"
        body = (
            f"## Summary\n"
            f"- endpoint: /classify\n"
            f"- input(text): `{text}`\n"
            f"- length: {len(text)}\n\n"
            f"## Exception\n"
            f"- type: {type(e).__name__}\n"
            f"- message: {str(e)}\n\n"
            f"## Traceback\n"
            f"```text\n{tb}\n```"
        )
        create_github_issue(title, body, logger)

        # (E) 사용자 응답은 심플하게
        return {"label": "Internal Server Error","score": -1}

    return {
        "label": label, "score": score
    }