from utils.emotion_detector import predict_emotion

text = "I am very happy today."

emotion = predict_emotion(text)

print("Input :", text)
print("Emotion :", emotion)