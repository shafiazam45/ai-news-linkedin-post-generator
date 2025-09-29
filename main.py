import os
import json
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain import LLMChain, PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Google Generative AI SDK
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in .env")
if not SERPAPI_API_KEY:
    raise RuntimeError("Set SERPAPI_API_KEY in .env")

# Configure Google SDK
genai.configure(api_key=GOOGLE_API_KEY)

# Collect available models
available_models = list(genai.list_models())
available_names = [m.name for m in available_models]

PREFERRED_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]

SELECTED_MODEL = None
for candidate in PREFERRED_MODELS:
    if candidate in available_names:
        SELECTED_MODEL = candidate
        break

if not SELECTED_MODEL:
    raise RuntimeError(
        f"No supported Gemini models found. Available: {available_names}"
    )

app = FastAPI(title="Demanual AI - News→LinkedIn Post", version="1.2")


class GenerateRequest(BaseModel):
    topic: str


class GenerateResponse(BaseModel):
    topic: str
    news_sources: List[str]
    linkedin_post: str
    image_suggestion: Optional[str] = None


# ---------- Tools ----------
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)


def create_search_tool():
    def run(query: str) -> str:
        return search.run(query)

    return Tool(
        name="web_search",
        description="Search the web for recent news & articles. Input is a search query string.",
        func=run,
    )


def create_gemini_llm():
    return ChatGoogleGenerativeAI(
        model=SELECTED_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )


# ---------- API Endpoints ----------
@app.post("/generate-post", response_model=GenerateResponse)
async def generate_post(payload: GenerateRequest):
    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must be non-empty")

    # Tools + LLM
    web_tool = create_search_tool()
    llm = create_gemini_llm()

    agent = initialize_agent(
        tools=[web_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

    # Search
    query = f"{topic} news - past week"
    try:
        search_text = web_tool.run(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Extract top URLs via SerpAPI JSON
    try:
        from serpapi import GoogleSearch

        params = {
            "engine": "google",
            "q": f"{topic} news",
            "api_key": SERPAPI_API_KEY,
            "num": "10",
        }
        serp = GoogleSearch(params)
        results = serp.get_dict()
        urls = []
        for r in results.get("organic_results", [])[:5]:
            link = r.get("link")
            if link:
                urls.append(link)
    except Exception:
        urls = []

    # Prompt
    prompt_text = """
You are a professional writer crafting a LinkedIn post. Given the TOPIC: "{topic}",
and the SEARCH_SUMMARY below, and optionally these SOURCE_URLS, write a short (3-6 sentences)
LinkedIn-style post in a professional and engaging tone that:
- Summarizes the most important developments,
- Mentions why it matters to professionals,
- Ends with a short call-to-action or reflection.

SEARCH_SUMMARY:
\"\"\"{search_summary}\"\"\"

SOURCE_URLS:
{source_urls}

Produce JSON with fields: "linkedin_post" and "image_suggestion" (URL or null).
Do not include any other text.
"""
    prompt = PromptTemplate.from_template(prompt_text)

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        resp = chain.run(
            {
                "topic": topic,
                "search_summary": search_text,
                "source_urls": "\n".join(urls),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    # Parse JSON
    json_text = None
    m = re.search(r"(\{[\s\S]*\})", resp)
    if m:
        json_text = m.group(1)
    else:
        json_text = json.dumps(
            {"linkedin_post": resp.strip(), "image_suggestion": None}
        )

    try:
        out = json.loads(json_text)
        linkedin_post = out.get("linkedin_post") or out.get("post") or resp.strip()
        image_suggestion = out.get("image_suggestion")
    except Exception:
        linkedin_post = resp.strip()
        image_suggestion = None

    return {
        "topic": topic,
        "news_sources": urls,
        "linkedin_post": linkedin_post,
        "image_suggestion": image_suggestion,
    }


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "service": "demanualai-news-linkedin-post",
        "using_model": SELECTED_MODEL,
    }


