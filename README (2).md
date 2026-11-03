# 💬 Emotion Classification with Attention

Multi-class emotion classifier (**joy, sadness, anger, fear, surprise, disgust**) built on the Google **GoEmotions** dataset, comparing recurrent models with attention against a fine-tuned transformer, and deployed as an interactive Gradio app that responds to the user's mood with a supportive message.

## ✨ Features
- GoEmotions labels (27 emotions) mapped to 6 core emotions (Ekman grouping)
- Embedding layer initialized with pre-trained **GloVe** vectors
- Four models compared: **LSTM**, **GRU**, **BiLSTM + Attention**, **DistilBERT** (HuggingFace Trainer)
- Evaluation: accuracy, macro F1, per-class precision/recall, confusion matrices
- **Attention heatmaps** showing which words drive each prediction
- **Gradio app**: dominant emotion, confidence bar chart, and an emotion-aware message
  - 😢 Sadness: comfort and a small tip
  - 😄 Joy: amplifies the positive mood
  - 😰 Fear/nervous: breathing and grounding exercise

## 📊 Results (test set)
| Model | Accuracy | Macro F1 |
|---|---|---|
| LSTM | _fill in_ | _fill in_ |
| GRU | _fill in_ | _fill in_ |
| BiLSTM + Attention | _fill in_ | _fill in_ |
| DistilBERT (fine-tuned) | _fill in_ | _fill in_ |

_Copy these from the comparison table at the end of the notebook._

## 🖼️ Screenshots
_Add `screenshots/app.png` (the Gradio app) and `screenshots/attention.png` (a heatmap), then reference them here:_
```
![App](screenshots/app.png)
![Attention](screenshots/attention.png)
```

## 🗂️ Project structure
```
├── notebooks/Emotion_Classification_with_Attention.ipynb   # full pipeline
├── app.py                                                  # Gradio app
├── requirements.txt
└── README.md
```

## 🚀 Run it
**1. Train:** open the notebook in Google Colab (GPU runtime) and run all cells.

**2. Run the app locally:**
```bash
git clone https://github.com/mahmoudeimo95/emotion-classification-attention.git
cd emotion-classification-attention
pip install -r requirements.txt
# Use the model hosted on the Hugging Face Hub (or a local folder)
MODEL_DIR=<your-hf-username>/emotion-distilbert python app.py
```
On Windows PowerShell: `$env:MODEL_DIR="<your-hf-username>/emotion-distilbert"; python app.py`

## 🧠 Method
1. Load GoEmotions (simplified) and map labels to 6 classes; drop neutral and mixed-class rows
2. Clean text, tokenize, build vocabulary, pad to fixed length (40)
3. Initialize the embedding layer with GloVe (100d)
4. Train LSTM, GRU and BiLSTM + additive attention with class-weighted loss (disgust is rare)
5. Fine-tune DistilBERT with the HuggingFace `Trainer`
6. Evaluate and visualize attention, then deploy

## 🛠️ Tech stack
Python · PyTorch · HuggingFace Transformers/Datasets · GloVe (gensim) · scikit-learn · Gradio · Matplotlib/Seaborn

## ⚠️ Notes
- The app gives general emotional-support messages. It is not a medical or mental-health tool.
- GoEmotions is Reddit-based, so the model may be less accurate on very different writing styles and on sarcasm.

## 👤 Author
**Mahmoud**: [GitHub](https://github.com/mahmoudeimo95)

## 📄 License
MIT
