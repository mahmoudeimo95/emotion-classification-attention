import os
import random

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "emotion_distilbert")  # local folder or HF Hub id
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
LABELS = [model.config.id2label[i] for i in range(model.config.num_labels)]

SUPPORT = {
    "sadness": {
        "emoji": "💙",
        "messages": [
            "I'm sorry you're feeling low. It's okay to feel this way, you don't have to push it away. Be gentle with yourself today. Tomorrow is a fresh start, and it can be better.",
            "That sounds heavy, and your feelings are valid. You've gotten through hard days before, and this one will pass too. Tomorrow will be better.",
            "Sadness shows that something mattered to you. Take your time. You're not alone in this, and brighter days are coming.",
        ],
        "tips": [
            "Write down one small thing that went okay today, even a tiny one.",
            "Reach out to someone you trust, even just to say hi.",
            "Drink some water, step outside for a few minutes, or play a song you love.",
            "Rest is productive too. Go easy on your to-do list today.",
        ],
    },
    "joy": {
        "emoji": "🎉",
        "messages": [
            "That's wonderful to hear! You deserve this happiness. Soak it all in!",
            "Love this energy! Moments like this are worth celebrating.",
            "So glad you're feeling great! Let this joy carry you through the whole day and beyond.",
        ],
        "tips": [
            "Share the good news with someone. Joy grows when it's shared!",
            "Write down what made you happy so you can revisit it on tougher days.",
            "Treat yourself to something small to celebrate the moment.",
            "Pay it forward: send a kind message to someone today.",
        ],
    },
    "fear": {
        "emoji": "🌿",
        "messages": [
            "Feeling nervous is normal, and it usually means something matters to you. Let's slow things down together for a moment.",
            "It's okay to feel anxious. You don't have to figure everything out right now. Just the next small step.",
        ],
        "tips": [
            "Breathe with me: in for 4 seconds, hold for 4, out for 6. Repeat 3 times.",
            "Grounding (5-4-3-2-1): name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
            "Ask yourself: what is ONE thing I can control right now? Start there.",
            "Ask yourself: what would I tell a friend who felt this way?",
        ],
    },
    "anger": {
        "emoji": "🔥",
        "messages": [
            "It sounds like something really got to you. Your frustration is valid.",
            "Anger often points to something that crossed a line for you. It's okay to feel it. Let's cool down before acting on it.",
        ],
        "tips": [
            "Pause before replying or reacting. Give it 10 slow breaths.",
            "Take a short walk or splash cold water on your face.",
            "Write out everything you feel, then decide what is actually worth acting on.",
        ],
    },
    "surprise": {
        "emoji": "✨",
        "messages": [
            "Wow, that sounds unexpected! Take a moment to let it sink in.",
            "Life certainly keeps things interesting! However you feel about it, you can handle it.",
        ],
        "tips": [
            "Give yourself a minute to process before making any decisions.",
            "Tell someone about it. Talking helps make sense of surprises.",
        ],
    },
    "disgust": {
        "emoji": "🍃",
        "messages": [
            "That sounds really unpleasant. It's fair to want distance from whatever caused it.",
            "Some things just rub us the wrong way, and that's valid. Let's reset your mood.",
        ],
        "tips": [
            "Step away from whatever caused it and change your environment for a bit.",
            "Do something refreshing: fresh air, a shower, or a favorite snack.",
        ],
    },
}

LOW_CONFIDENCE = "I'm not totally sure how you're feeling, and that's okay. Feel free to tell me a bit more. 🙂"


def build_message(emotion, confidence):
    if emotion is None:
        return ""
    if confidence < 0.40:
        return LOW_CONFIDENCE
    s = SUPPORT[emotion]
    return f"### {s['emoji']} {random.choice(s['messages'])}\n\n**💡 Try this:** {random.choice(s['tips'])}"


def classify(text):
    if not text or not text.strip():
        return {}, "Type how you're feeling and I'll tell you what I sense.", None, 0.0
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top = max(scores, key=scores.get)
    return scores, build_message(top, scores[top]), top, scores[top]


def another(emotion, confidence):
    return build_message(emotion, confidence or 0.0)


with gr.Blocks(title="Emotion Classifier") as demo:
    gr.Markdown("# 💬 Emotion Classification with Attention\nWrite how you feel in your own words.")
    emotion_state = gr.State(None)
    conf_state = gr.State(0.0)
    inp = gr.Textbox(lines=3, label="Your text", placeholder="e.g. I have a big interview tomorrow and I can't stop worrying...")
    with gr.Row():
        btn = gr.Button("Analyze", variant="primary")
        again = gr.Button("Give me another message")
    with gr.Row():
        chart = gr.Label(num_top_classes=6, label="Confidence across emotions")
        msg = gr.Markdown()
    outputs = [chart, msg, emotion_state, conf_state]
    btn.click(classify, inp, outputs)
    inp.submit(classify, inp, outputs)
    again.click(another, [emotion_state, conf_state], msg)

if __name__ == "__main__":
    demo.launch()
