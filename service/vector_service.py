""" app to search through the session data using the vector embeddings
Used from  https://github.com/gloveboxes/Semantic-Search-with-OpenAI-Embeddings-and-Functions (2023)
"""

import os
import openai
import pandas as pd
from typing import List
from pydantic import BaseModel
from openai.embeddings_utils import get_embedding, cosine_similarity
from fastapi import FastAPI, UploadFile, Response, status, Request
import uvicorn
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY = 
RESOURCE_ENDPOINT = 
ENGINE_NAME = os.getenv("AZURE_ENGINE_NAME")
openai.api_type =   os.getenv("AZURE_OPENAI_API_TYPE") 
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VER")

df_sessions = pd.read_json("../prep/output/clean_embeddings.json")

app = FastAPI()


class Session(BaseModel):
    videoId: str
    start: str
    speaker: str
    title: str
    similarities: float
    summary: str
    description: str


@app.get("/search", response_model=List[Session], status_code=200)
async def create_upload_file(
    query: str, top_n: int = 6, dedup: bool = True
) -> List[Session]:
    global df_sessions
    """Search the documents using the user query and return the top_n results"""
    embedding = get_embedding(query, engine=ENGINE_NAME)
    df_sessions["similarities"] = df_sessions["ada_v2"].apply(
        lambda x: cosine_similarity(x, embedding)
    )

    # Checking for NaN values in similarities and drop them
    df_sessions = df_sessions.dropna(subset=["similarities"])
    if dedup:
        res = df_sessions.sort_values("similarities", ascending=False).drop_duplicates(
            subset=["videoId"]
        )

    else:
        res = df_sessions.sort_values("similarities", ascending=False)

    res = (
        res.head(top_n).drop(columns=["ada_v2"]).drop(columns=["text"]).fillna("")
    ).to_dict("records")

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5500)
