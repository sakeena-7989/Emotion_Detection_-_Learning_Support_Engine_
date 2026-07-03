"""
Safe Emotion Detector (Fallback Version)
This version does NOT require ML model files.
Used until BiLSTM model is properly fixed.
"""

def predict_emotion(text):
    text = str(text).lower()

    if any(word in text for word in ["happy", "great", "good", "excellent", "love", "amazing"]):
        return "😊 Happy"

    elif any(word in text for word in ["sad", "cry", "upset", "depressed", "rotten", "shitty"]):
        return "😢 Sad"

    elif any(word in text for word in ["angry", "hate", "mad", "annoyed"]):
        return "😠 Angry"

    elif any(word in text for word in ["fear", "scared", "afraid", "nervous", "uncomfortable"]):
        return "😨 Fear"

    elif any(word in text for word in ["surprised", "wow", "shocked"]):
        return "😲 Surprise"

    return "😐 Neutral"