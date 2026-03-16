import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
# from google import genai
# from dotenv import load_dotenv
# import os

# # This loads the variables from .env into your system environment
# load_dotenv()


def create_embeddings(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()['embeddings']
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    response = r.json()
    print(response)
    return response

# def inference(prompt):
#     # The client will now automatically find 'GOOGLE_API_KEY'
#     client = genai.Client()
#     r = client.models.generate_content(
#         model="gemini-3-flash-preview",
#         contents=prompt,
#     )
#     response = r.json()
#     return response


df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a query: ")
question_embedding = create_embeddings([incoming_query])[0]
# print(question_embedding)

#Find similarity of question_embedding with other embeddings.
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[['number','title','text']])

prompt = f'''I am a teaching web development in my Web development course. Here are video chunks containing video number, video title, start time in seconds, end time in seconds, the text at that time:
{new_df[['number','title','start','end','text']].to_json(orient="records")}
------------------------------
"{incoming_query}"
 User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, its just for you) where and how much content is taught in which video (in which video and what timestamp) and guide the user to go to that particular video. If user asks unreleated  question, tell him that you can only answer a question related to the content in the video chunks and you don't have any knowledge beyond that.
'''
with open("prompt.txt","w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(f"Index: {index}, Number: {item['number']}, Title: {item['title']}, Text: {item['text']}, Start: {item['start']}, End: {item['end']}")