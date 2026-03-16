import whisper
import json
model = whisper.load_model("large-v2")
result = model.transcribe(audio = "audios/1_Installing VS Code & How Websites Work Sigma Web Development Course - Tutorial #1 [fgh].mp4.mp3",
                          language = "hi",
                          task = "translate",
                          word_timestamps=False)
# print(result["segments"])
chunks = []
for segment in result["segments"]:
    chunk = {
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    }
    chunks.append(chunk)

print(chunks)
# with open("output.json", "w") as f:
#     json.dump(f, result)