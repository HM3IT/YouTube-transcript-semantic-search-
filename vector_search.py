""" app to search through the session data using the vector embeddings"""

import os
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
ENGINE_NAME = os.getenv("AZURE_ENGINE_NAME")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2023-05-15"


# load the session data from csv file
df_sessions = pd.read_csv("master_embeddings.csv")
# convert the embedding column from string to list
df_sessions["ada_v2"] = df_sessions["ada_v2"].apply(lambda x: eval(x))


def search_docs(df, user_query, top_n=3):
    """search the documents using the user query and return the top_n results"""
    try:
        embedding = get_embedding(
            user_query,
            # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
            engine=ENGINE_NAME,
        )
        df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    except openai.error.InvalidRequestError as e:
        print(f"Invalid Request Error: {e}")
        raise
    res = (
        df.sort_values("similarities", ascending=False)
        # dedup the results
        .drop_duplicates("title").head(top_n)
    )

    return r.text


while True:
    input_text = input("Enter your query: ")
    if input_text is None or input_text == "":
        continue
    res = search_docs(df_sessions, input_text, top_n=6)
    print(res[["title", "speaker", "videoId", "similarities"]])
