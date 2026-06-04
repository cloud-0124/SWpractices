import logging
import traceback

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import LOW_CONFIDENCE_THRESHOLD, MODEL_MODE
from app.issue import create_github_issue
from app.model_loader import get_model_info
from app.retrain_issue import update_issue_state
from app.spam import check_spam_ml_canary, check_spam_rules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | "
           "%(filename)s:%(lineno)d (%(funcName)s) | "
           "%(message)s",
)
logger = logging.getLogger("spamcheck")

app = FastAPI(title="SpamCheck Web")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


class ClassifyRequest(BaseModel):
    text: str


@app.post("/classify")
async def classify(payload: ClassifyRequest):
    text = payload.text
    serving_model = "rules"

    logger.info(f"CALL /classify | text='{text}' | len={len(text)}")

    try:
        if MODEL_MODE == "ml":
            label, score, serving_model = check_spam_ml_canary(text)
            update_issue_state(text, label, score, LOW_CONFIDENCE_THRESHOLD)
        else:
            label, score = check_spam_rules(text)

        logger.info(f"OK /classify | label={label} score={score}")

    except Exception as e:
        logger.exception(
            f"FAIL /classify | text='{text}' | error={type(e).__name__}: {e}"
        )

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
        return {"label": "Internal Server Error", "score": -1}

    return {
        "label": label,
        "score": score,
        "serving_model": serving_model,
        "model_info": get_model_info(serving_model),
    }
