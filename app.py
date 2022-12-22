
from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chatbot import predict
from quiz import predict_quiz
from speech_emotion_recognition import predict_speech_emotion
from facial_emotion_recognition import predict_facial_emotion

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the API!"}


@app.post("/chatbot")
async def func(request: Request):
    data = await request.json()
    print(data['question'])
    reply = predict(data['question'])
    return {"reply": reply}


@app.post("/quiz")
async def func(request: Request):
    data = await request.json()
    print(data['answers'])
    reply = predict_quiz(data['answers'])
    return {"reply": reply}


@app.post("/emotion-speech")
async def func():
    # data = await request.json()
    # print(data['answers'])
    emotion = predict_speech_emotion()
    text_list = ['neutral', 'calm', 'happy',
                    'sad', 'angry', 'fearful', 'disgust', 'surprised']

    if emotion == 0:
        return text_list[0]
    if emotion == 1:
        return text_list[1]
    elif emotion == 2:
        return text_list[2]
    elif emotion == 3:
        return text_list[3]
    elif emotion == 4:
        return text_list[4]
    elif emotion == 5:
        return text_list[5]
    elif emotion == 6:
        return text_list[6]

# text_list = ['Angry', 'Disgust', 'Fear',
                    #  'Happy', 'Neutral', 'Sad', 'Surprise']


@app.post("/emotion-facial")
async def func():
    # data = await request.json()
    # print(data['answers'])
    emotion = predict_facial_emotion()
    text_list = ['Angry', 'Disgust', 'Fear',
                    'Happy', 'Neutral', 'Sad', 'Surprise']

    if emotion == 0:
        return text_list[0]
    if emotion == 1:
        return text_list[1]
    elif emotion == 2:
        return text_list[2]
    elif emotion == 3:
        return text_list[3]
    elif emotion == 4:
        return text_list[4]
    elif emotion == 5:
        return text_list[5]
    elif emotion == 6:
        return text_list[6]