# How to use RAG Based AI Teaching Assistant on you own data
## Step 1 - Collect Videos
Move all video files in the video folder

## Step 2 - Convert to Mp3
Convert all video files to Mp3 by running video_to_mp3

## Step 3 - Convert Mp3 to json
Convert all mp3 files to json by running mp3_to_json

## Step 4 - Convert the json files to vectors
use the file create_embedding_of_json to convert the json file  to a dataframe with embeddings and save it as a joblib pickle

## Step 5 Prompt Generatoin and response to LLM
Read the joblib file and load it into the memory.  Then create a relevant prompt as per the user query and feed it to the LLM.