from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from collections import Counter
import emoji
#from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

# ---------- 1) Forwarded / probable spam detector ----------
def detect_forwarded_messages(df):
    """
    Returns DataFrame of messages flagged as forwarded/probable-spam.
    Heuristics used:
      - message contains 'forwarded' (case-insensitive) OR
      - message starts with typical forward markers OR
      - contains many URLs (>=2) OR
      - length very short + many links/mentions
    """
    temp = df.copy()
    temp = temp[~temp['message'].isna()]
    def is_forward(msg):
        m = str(msg).lower()
        if 'forwarded' in m:
            return True
        # WhatsApp sometimes has '→' or '↪' or 'Fwd'
        if m.strip().startswith(('fwd', 'fw', '→', '↪')):
            return True
        # many links heuristic
        url_count = len(extract.find_urls(str(msg)))
        if url_count >= 2:
            return True
        return False

    temp['is_forwarded_suggest'] = temp['message'].apply(is_forward)
    return temp[temp['is_forwarded_suggest'] == True][['user','date','message']].reset_index(drop=True)


# ---------- 2) Duplicate / near-duplicate detector ----------
def detect_duplicate_messages(df, similarity_threshold=0.85):
    """
    Returns groups of duplicate or near-duplicate messages.
    Approach:
      - preprocess (lower, strip punctuation)
      - TF-IDF on unique messages
      - compute cosine similarity matrix and cluster pairs above threshold
    Output:
      DataFrame with columns: user, date, message, duplicate_group (int)
    """
    temp = df.copy()
    temp = temp[~temp['message'].isna()].reset_index(drop=True)
    # simple normalize
    def normalize(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'[^a-z0-9\u0900-\u097F\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    temp['norm'] = temp['message'].apply(normalize)

    # dedupe exact duplicates fast
    dup_mask = temp['norm'].duplicated(keep=False)
    if dup_mask.sum() == 0:
        # no exact duplicates; proceed to near-duplicate detection
        pass

    unique_texts = temp['norm'].unique().tolist()
    if len(unique_texts) < 2:
        return pd.DataFrame()  # nothing to do

    vect = TfidfVectorizer(min_df=1).fit_transform(unique_texts)
    sim = cosine_similarity(vect)
    # build groups using simple agglomerative-like union-find of pairs > threshold
    n = len(unique_texts)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb:
            parent[rb]=ra

    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j] >= similarity_threshold:
                union(i,j)

    groups = {}
    for idx, txt in enumerate(unique_texts):
        root = find(idx)
        groups.setdefault(root, []).append(txt)

    # map messages to group ids
    group_map = {}
    for gid, (root, texts) in enumerate(groups.items(), start=1):
        for t in texts:
            group_map[t] = gid

    temp['duplicate_group'] = temp['norm'].map(group_map).fillna(0).astype(int)
    result = temp[temp['duplicate_group'] != 0][['user','date','message','duplicate_group']].sort_values('duplicate_group')
    return result.reset_index(drop=True)


# ---------- 3) Simple extractive summarizer for long messages ----------
def summarize_long_messages(df, max_chars=400, num_sentences=2):
    """
    For messages longer than `max_chars`, produce an extractive summary
    by scoring sentences using TF-IDF and picking top `num_sentences`.
    Returns DataFrame: user, date, message, summary
    """
    temp = df.copy()
    temp = temp[~temp['message'].isna()].reset_index(drop=True)

    records = []
    for _, row in temp.iterrows():
        text = str(row['message']).strip()
        if len(text) <= max_chars:
            continue
        # split into sentences (simple split on punctuation)
        sents = re.split(r'(?<=[.!?।])\s+', text)
        sents = [s.strip() for s in sents if len(s.strip())>10]
        if len(sents) == 0:
            continue
        # if few sentences, return first 1-2
        if len(sents) <= num_sentences:
            summary = " ".join(sents[:num_sentences])
            records.append({'user': row['user'],'date': row['date'],'message': text,'summary': summary})
            continue

        # score sentences using TF-IDF (per-document)
        vect = TfidfVectorizer().fit_transform(sents)
        scores = np.array(vect.sum(axis=1)).ravel()
        top_idx = scores.argsort()[::-1][:num_sentences]
        # preserve original order
        top_idx_sorted = sorted(top_idx.tolist())
        summary = " ".join([sents[i] for i in top_idx_sorted])
        records.append({'user': row['user'],'date': row['date'],'message': text,'summary': summary})

    return pd.DataFrame(records)


# ---------- 4) Convenience: generate cleanup suggestions table ----------
def generate_cleanup_suggestions(df):
    """
    Returns a single DataFrame with suggested cleanup actions:
      - type: Forwarded, Duplicate, LongMessage
      - suggestion: text describing action (e.g., 'Consider removing forwarded message')
    """
    fw = detect_forwarded_messages(df)
    fw['issue_type'] = 'Forwarded / Probable Spam'
    fw['suggestion'] = 'Review / remove forwarded message'

    dup = detect_duplicate_messages(df)
    if not dup.empty:
        # for duplicates, pick representative message per group
        dup_group_summary = dup.groupby('duplicate_group').agg({
            'user': lambda x: ", ".join(set(x)),
            'date': 'min',
        }).reset_index()
        # join back a short message sample
        samples = dup.groupby('duplicate_group')['message'].first().reset_index()
        dup_summary = dup_group_summary.merge(samples, on='duplicate_group')
        dup_summary = dup_summary.rename(columns={'message':'message_sample'})
        dup_summary = dup_summary.rename(columns={'duplicate_group':'group_id'})
        dup_df = dup_summary[['user','date','message_sample','group_id']].rename(columns={'message_sample':'message'})
        dup_df['issue_type'] = 'Duplicate/Near-Duplicate'
        dup_df['suggestion'] = 'Consider keeping 1 copy per group_id'
    else:
        dup_df = pd.DataFrame(columns=['user','date','message','issue_type','suggestion'])

    long_df = summarize_long_messages(df)
    if not long_df.empty:
        long_df = long_df.rename(columns={'summary':'suggestion'})
        long_df['issue_type'] = 'Long Message'
        # suggestion column already has summary; prefix for clarity
        long_df['suggestion'] = 'Suggested summary: ' + long_df['suggestion']
        long_df = long_df[['user','date','message','issue_type','suggestion']]
    else:
        long_df = pd.DataFrame(columns=['user','date','message','issue_type','suggestion'])

    # unify columns
    fw = fw[['user','date','message','issue_type','suggestion']]
    dup_df = dup_df[['user','date','message','issue_type','suggestion']]
    combined = pd.concat([fw, dup_df, long_df], ignore_index=True, sort=False).reset_index(drop=True)
    return combined


extract = URLExtract()#links nikalne ke liye

# Emoji sentiment mapping (simple version)
emoji_sentiment = {
    "😀": 2.0, "😃": 2.0, "😄": 2.0, "😁": 2.0, "😆": 2.0, "😅": 1.5,
    "😂": 2.5, "🤣": 2.5, "😊": 2.0, "😇": 2.0, "🙂": 1.5, "😉": 1.5,
    "😍": 3.0, "😘": 2.5, "❤️": 3.0, "👌": 2.0, "😎": 2.0, "🤩": 2.5,
    "😢": -2.0, "😭": -2.5, "😞": -2.0, "😔": -2.0, "☹️": -2.0,
    "😡": -3.0, "🤬": -3.0, "😠": -2.5, "🤢": -2.0, "🤮": -2.5,
}

# ----- Sentiment Analyzer (English + Hinglish) -----
sia = SentimentIntensityAnalyzer()

# Hinglish / slang words ka custom sentiment lexicon
hinglish_lexicon = {
    "mast": 2.5, "badiya": 2.5, "badiyaaa": 2.5, "sahi": 1.5, "sahihai": 2.0, "sahih": 2.0, "acha": 1.5, "accha": 1.5, "bohotaccha": 3.0, "mazaaagaya": 3.0, "mazaaagya": 3.0, "op": 2.0, "jhakas": 2.5, "solid": 2.0,
    "bakwas": -2.5, "ghatiya": -3.0, "faltu": -2.0, "bura": -2.0, "bkwas": -2.5, "bewakoof": -2.0, "gussa": -2.0, "gandi": -2.5, "sad": -2.0,
    "sceneon": 2.2, "fullpower": 2.5, "jhakasbro": 2.6, "dhaasu": 2.3, "tejbanda": 1.7, "bohotzabardast": 3.0, "properop": 2.8, "fullrespect": 2.7, "shandar": 2.5, "kyabaathai": 2.6, "yaarkamaal": 2.4, "bhaiOPhotum": 3.0, "cutevibe": 1.9, "fullentertainment": 2.6, "moodfreshhogaya": 2.5, "dilkhushhogaya": 3.0,
    "bhaisuper": 2.6, "bhaitussigreat": 2.8, "motimotivation": 2.3, "ekdumzabardast": 2.8, "topgamer": 2.4, "bhaiesportslevel": 2.8, "perfectacting": 2.5, "cuteattitude": 2.0, "talentdikhgaya": 2.6, "kingentry": 2.3, "queenentry": 2.3, "supersebhiupar": 3.0, "dilsesalute": 2.7, "cutereel": 1.7,
    "hardvibe": 2.4, "hardpunchline": 2.3, "jhakasdance": 2.4, "killerattitude": 2.3, "toppersonality": 2.4, "realsupportive": 2.4, "valuemilgayi": 2.5, "originalbanda": 2.0, "brandhotum": 2.4, "insaneskill": 2.6, "insaneedit": 2.5, "maakasamop": 2.8, "bohotpyaralagrha": 2.2, "bohotpyarialagrhi": 2.2, "nocap": 2.0,
    "hardworkpaid": 2.7, "mindsetkiller": 2.3, "energeticbanda": 2.0, "puretalent": 2.4, "kadak": 2.5, "jabardast": 2.8, "zabardast": 2.8, "bohotsahi": 2.8, "lit": 2.2, "fire": 2.2, "amazingbro": 2.6, "chillvibes": 2.2, "jhakkashai": 2.6, "bohothard": 3.0, "heavydriver": 2.4,
    "legendbhai": 2.6, "opbolte": 2.8, "topclass": 2.7, "eknumber": 2.4, "kamalka": 2.6, "bhaikamaal": 2.6, "overpowered": 2.7, "noice": 2.0, "banger": 2.4, "classystuff": 2.3, "hatsoffbro": 2.8, "respectmilgayi": 2.8, "smartmove": 2.2, "realone": 2.1,
    "cutelagrhi": 1.8, "cutelagrame": 1.8, "diljeetliya": 3.2, "bhaifirecontent": 2.9, "killerlook": 2.2, "opgameplay": 2.7, "godlevel": 3.2, "nextlevel": 3.1, "tagda": 2.4, "bohotpyara": 2.6, "paisavasool": 2.5, "perfectbro": 2.7, "rocking": 2.2, "cutesmile": 1.9,
    "faltuacting": -2.3, "faltuopinion": -2.2, "bhaibakwas": -2.6, "burisoch": -2.0, "cheapsoch": -2.3, "gandagimatchfaila": -2.1, "hatevibe": -2.3, "cringe": -2.3, "cringereel": -2.4, "cringebanda": -2.5, "cringereply": -2.4, "zerotalent": -2.2, "zerologic": -2.4,
    "absolutelybakwas": -2.8, "characterdheela": -2.5, "jhagdawalaattitude": -2.0, "demean": -2.0, "fakesmile": -2.1, "jealousnature": -2.3, "gandiharkate": -2.6, "faltubaate": -2.3, "pagalpankihad": -2.5, "toxicreply": -2.7, "toxicfan": -2.6, "kaamzero": -2.2, "badtameezbanda": -2.5,
    "cheapacting": -2.4, "dhokebaaz": -2.6, "uselesscheez": -2.6, "abusivelanguage": -3.0, "worstever": -3.0, "patheticlevel": -2.9, "nonsensebaat": -2.3, "bakwaascontent": -2.6, "irritatinghai": -2.2, "insecurebanda": -2.1, "insulthogayi": -2.3, "sadlife": -2.1, "faltureel": -2.2

}



# In words ko VADER lexicon me merge kar do
sia.lexicon.update(hinglish_lexicon)


def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
       df= df[df['user'] == selected_user]

    #kitne message he
    num_messages = df.shape[0]

    #words ke liye list bana li
    words = []
    for message in df['message']:
        words.extend(message.split())#message ko words me split kar liya

    num_media_messages=df[df['message']=='<Media omitted>\n'].shape[0]

    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))#link nikal dega alag se

    return num_messages, len(words),num_media_messages,len(links)

def most_busy_users(df):
    x=df['user'].value_counts().head(10)
    df=round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'index':'name','user':'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y=[]
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return ' '.join(y)

    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message']= temp['message'].apply(remove_stop_words)
    df_wc=wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]

    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']
    words=[]
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df=pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]
    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]

    timeline=df.groupby(['year','month_num','month']).count()['message'].reset_index()

    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+"-"+str(timeline['year'][i]))

    timeline['time']=time

    return timeline

def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]

    daily_timeline=df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df= df[df['user'] == selected_user]

    user_heatmap=df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_analysis(selected_user, df):
    # user filter
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # group notifications / media remove
    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    # VADER compound score nikalna (English + Hinglish + custom lexicon)
    def get_compound(text):
        text = str(text)
        base = sia.polarity_scores(text)['compound']

        # emoji score add karo
        extra = 0
        for ch in text:
            if ch in emoji_sentiment:
                extra += emoji_sentiment[ch] / 10.0  # scale down

        score = base + extra
        # clamp between -1 & 1
        if score > 1:
            score = 1
        if score < -1:
            score = -1
        return score

    temp['compound'] = temp['message'].apply(get_compound)

    # score ko label me convert karo
    def to_label(c):
        if c >= 0.05:
            return 'Positive'
        elif c <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    temp['sentiment'] = temp['compound'].apply(to_label)

    # pie chart ke liye counts
    counts = temp['sentiment'].value_counts()

    # table ke liye message + sentiment bhejo
    return temp[['message', 'sentiment']], counts

# very simple abuse / profanity detection
abuse_words = {
    "stupid", "idiot", "gandu", "harami", "chutiya", "chutiye",
    "bhosdi", "bhosdike", "bkl", "madarchod", "mc", "bc",
    "bhenchod", "bhen ch*d", "bsdk" , "gaand", "kutta", "kutti"

}

def abuse_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    rows = []
    for _, row in temp.iterrows():
        text = str(row['message']).lower()
        found = [w for w in abuse_words if w in text]
        if found:
            rows.append({
                'user': row['user'],
                'message': row['message'],
                'abusive_terms': ", ".join(set(found))
            })

    abuse_df = pd.DataFrame(rows)
    return abuse_df

def user_sentiment_summary(df):
    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def get_compound(text):
        return sia.polarity_scores(str(text))['compound']

    temp['compound'] = temp['message'].apply(get_compound)

    summary = temp.groupby('user')['compound'].mean().sort_values()
    return summary

def sentiment_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def get_compound(text):
        return sia.polarity_scores(str(text))['compound']

    temp['compound'] = temp['message'].apply(get_compound)

    # assuming df['date'] is datetime
    daily = temp.groupby('only_date')['compound'].mean().reset_index()
    return daily

def user_sentiment_summary(df):
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    summary = temp.groupby('user')['sentiment'].value_counts().unstack().fillna(0)
    return summary

def topic_modeling(selected_user, df, num_topics=5, num_words=8):

    # 1) User filter
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[~temp['message'].isna()]

    if temp.empty:
        return pd.DataFrame()

    # 2) Hinglish stopwords
    stop_words = []
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = f.read().split()
    except:
        stop_words = []

    extra_stop = ['media', 'omitted', 'deleted', 'image', 'video', 'photo']
    stop_words = list(set(stop_words + extra_stop))
    stop_words = set([w.strip().lower() for w in stop_words])

    import re
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = ''.join(ch if not emoji.is_emoji(ch) else ' ' for ch in text)
        text = re.sub(r'[^a-zA-Z\u0900-\u097F\s]', ' ', text)
        tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
        return ' '.join(tokens)

    temp['clean'] = temp['message'].apply(clean_text)
    docs = temp['clean'][temp['clean'].str.len() > 0].tolist()

    if len(docs) < 3:
        return pd.DataFrame()

    n_topics = min(num_topics, len(docs))

    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words=list(stop_words)  # ❗ 2nd layer stopword filter
    )
    X = vectorizer.fit_transform(docs)

    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_model.fit(X)

    feature_names = vectorizer.get_feature_names_out()

    topics_data = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[::-1][:num_words]
        words = [feature_names[i] for i in top_indices if feature_names[i] not in stop_words]  # ❗ final filter
        topics_data.append({
            'Topic #': topic_idx + 1,
            'Top words / phrases': ", ".join(words)
        })

    return pd.DataFrame(topics_data)

def most_tagged_person(selected_user, df):
    """
    Chat me '@name' ya '@username' jitni baar aaye,
    unko count karke sabse zyada tagged log return karega.
    """

    # 1) User ke hisaab se filter
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # 2) Group notification / media hatao
    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[~temp['message'].isna()]

    tags = []

    # 3) Har message me @tags dhundo
    for msg in temp['message']:
        text = str(msg)
        # @ke baad jo bhi word aata hai (space / comma tak)
        found = re.findall(r'@([^\s,.:;!?]+)', text)
        tags.extend(found)

    # 4) Agar koi tag mila hi nahi
    if not tags:
        return pd.DataFrame(columns=['tag', 'count'])

    # 5) Count karo
    tag_counts = Counter(tags).most_common()
    tagged_df = pd.DataFrame(tag_counts, columns=['tag', 'count'])

    return tagged_df

def chat_awards(df):

    # remove meta ai + system/group notifications
    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[~temp['user'].str.contains("Meta", case=False, na=False)]
    temp = temp[temp['message'] != '<Media omitted>\n']

    # Users list
    users = temp['user'].unique().tolist()

    # 1️⃣ Most Supportive → highest positive sentiment messages
    from collections import defaultdict
    supportive = defaultdict(int)
    for _, r in temp.iterrows():
        if 'sentiment' in temp.columns:
            if r['sentiment'] == "Positive":
                supportive[r['user']] += 1

    supportive_winner = max(supportive, key=supportive.get) if supportive else None

    # 2️⃣ Funniest → max laughing emojis
    laugh_emojis = ["😂", "🤣", "😆", "😹"]
    funny = defaultdict(int)
    for _, r in temp.iterrows():
        msg = str(r['message'])
        for ch in msg:
            if ch in laugh_emojis:
                funny[r['user']] += 1

    funniest_winner = max(funny, key=funny.get) if funny else None

    # 3️⃣ Silent Reader → sabse kam messages (but present)
    counts = temp['user'].value_counts()
    silent_reader = counts.idxmin() if len(counts) > 0 else None

    return {
        "Most Supportive": supportive_winner,
        "Funniest": funniest_winner,
        "Silent Reader": silent_reader
    }

def fight_detector(df):
    """
    Detect heated arguments based on:
    - negative sentiment streak
    - fast replies (≤ 10 mins)
    - ≥ 2 users involved
    """

    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[~temp['user'].str.contains("Meta", case=False, na=False)]
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[~temp['sentiment'].isna()]  # sentiment must exist

    fights = []
    window = []   # store recent messages in window

    for i in range(len(temp)):
        row = temp.iloc[i]
        window.append(row)

        # window time check → keep only last 10 minutes
        start_time = window[0]['date']
        while (row['date'] - start_time).seconds / 60 > 10:
            window.pop(0)
            if len(window) > 0:
                start_time = window[0]['date']

        # condition check
        sentiments = [w['sentiment'] for w in window]
        users = set([w['user'] for w in window])

        negatives = sentiments.count("Negative")
        if negatives >= 3 and len(users) >= 2:
            fights.append({
                "Time Window Start": window[0]['date'],
                "Time Window End": window[-1]['date'],
                "Involved Users": ", ".join(users),
                "Messages Count": len(window),
                "Negative Msg Count": negatives
            })
            window.clear()   # avoid double detection

    return pd.DataFrame(fights)

def anger_diffusion_map(df):
    """
    Track how negative sentiment spreads across users:
    - find negative streaks
    - record order in which users contribute negativity
    - produce a timeline of diffusion
    """

    temp = df.copy()
    temp = temp[temp['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[~temp['sentiment'].isna()]
    temp = temp.reset_index(drop=True)

    diffusion_events = []

    window = []  # sliding window of recent messages (e.g., last 10 min)
    for i in range(len(temp)):
        row = temp.iloc[i]
        window.append(row)

        # keep only last 15 minutes
        start = window[0]['date']
        while (row['date'] - start).total_seconds() / 60 > 15:
            window.pop(0)
            if len(window) > 0:
                start = window[0]['date']

        # collect negative messages only
        negatives = [w for w in window if w['sentiment'] == "Negative"]
        if len(negatives) < 2:
            continue  # diffusion needs at least 2 users

        # involved unique users in negative chain
        users = list(dict.fromkeys([w['user'] for w in negatives]))

        if len(users) >= 2:
            diffusion_events.append({
                "Start Time": negatives[0]['date'],
                "End Time": negatives[-1]['date'],
                "Duration (mins)": round((negatives[-1]['date'] - negatives[0]['date']).total_seconds() / 60, 2),
                "Users in Order": " → ".join(users),
                "Negative Messages Count": len(negatives)
            })

    return pd.DataFrame(diffusion_events)



# def sentiment_analysis(selected_user, df):
#     # selected_user ke hisaab se filter
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#
#     # group notification / media remove
#     temp = df.copy()
#     temp = temp[temp['user'] != 'group_notification']
#     temp = temp[temp['message'] != '<Media omitted>\n']
#
#     # polarity nikalna
#     def get_polarity(text):
#         return TextBlob(text).sentiment.polarity
#
#     temp['polarity'] = temp['message'].apply(get_polarity)
#
#     # polarity ko label me convert karo
#     def label(p):
#         if p > 0.1:
#             return 'Positive'
#         elif p < -0.1:
#             return 'Negative'
#         else:
#             return 'Neutral'
#
#     temp['sentiment'] = temp['polarity'].apply(label)
#
#     # count for pie chart
#     counts = temp['sentiment'].value_counts()
#
#     # table ke liye sirf message + sentiment bhej do
#     return temp[['message', 'sentiment']], counts