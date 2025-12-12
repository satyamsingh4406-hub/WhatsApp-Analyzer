# 📊 WhatsApp Chat Analyzer (NLP + ML Powered)

A modern WhatsApp Chat Analysis Tool built using **Python, Streamlit, NLP, and Machine Learning**.  
This project extracts deep insights from WhatsApp exported chats, including **sentiment trends, message behavior, topics, activity patterns, anger diffusion, fights, toxicity detection, and more.**

---

## 🚀 Features

### 📈 1. Chat Statistics  
- Total messages  
- Total words  
- Media & links shared  
- User activity breakdown  

---

### 🗓️ 2. Timeline Analysis  
- Monthly timeline  
- Daily timeline  
- Weekly activity heatmap  
- Most active days & months  

---

### 🔤 3. Text & Emoji Insights  
- WordCloud (with Hinglish stopwords)  
- Most common words  
- Emoji usage statistics  

---

### 😊 4. Sentiment Analysis (English + Hinglish)  
- VADER sentiment engine  
- Custom Hinglish sentiment lexicon  
- Emoji sentiment boosting  
- Sentiment timeline  
- User-wise average sentiment visualization  

---

### 🚫 5. Abuse / Toxicity Detection  
- Detects offensive / abusive messages  
- Identifies abusive terms  
- Flags toxic users and messages  

---

### 🎯 6. Topic Modeling (Hinglish Compatible)  
- TF-IDF + NMF based topic extraction  
- Removes emojis, links, mentions  
- Multi-word phrase detection  
- Shows top topics in chat  

---

### 🏷️ 7. Mentions & Tag Analysis  
- Detects `@username` mentions  
- Ranks most-tagged people  
- Visual breakdown  

---

### 🏆 8. Chat Awards (Auto-Generated)  
- **Most Supportive User**  
- **Funniest User** (laugh emojis)  
- **Silent Reader** (least active)  

---

### 🔥 9. Fight / Argument Detector  
Detects arguments based on:  
- Negative sentiment spikes  
- Fast replies (≤ 10 minutes)  
- Multi-user involvement  
- Returns fight window, users, and intensity  

---

### 🌋 10. Anger Diffusion Map  
Shows how **anger spreads** across users:
- Negative chain starter  
- Order of users who amplify negativity  
- Duration of negative wave  
- Number of negative messages in chain  

---

### ⏳ 11. Message Lifespan Detector  
Tracks:  
- When message was posted  
- First reply time  
- Last reply time  
- How long message remained “active”  
- Reply count  
- Identifies “dead messages”  

---

## 🛠️ Tech Stack

**Frontend:** Streamlit  
**Backend:** Python  
**NLP:** NLTK, VADER, emoji, TF-IDF  
**ML:** NMF Topic Modeling  
**Visualization:** Matplotlib, Seaborn  
**Utilities:** URLEXTRACT, Pandas  

---

## 📦 Installation

```
pip install -r requirements.txt
```

---

## ▶️ Run the App

```
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                 # Streamlit UI
├── helper.py              # All NLP + ML logic
├── preprocessor.py        # Chat preprocessing
├── stop_hinglish.txt      # Custom stopwords
├── requirements.txt
└── README.md
```

---

## 📥 Input Format

Export WhatsApp chat as:

```
WhatsApp → 3 dots → More → Export chat → Without media
```

Upload `.txt` file in the Streamlit UI.

---

## 💡 Future Enhancements
- Chat summarization (LLM-powered)  
- Sentiment per topic  
- Toxicity severity scoring  
- Conversation tree visualization  

---

## ⭐ Show Your Support  
If you like this project, consider giving it a **star ⭐ on GitHub!**

---

Made with ❤️ using Python & Streamlit.
