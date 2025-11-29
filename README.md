# Day 04: Sentiment Analysis with NLP 📊💬

## 📝 Overview

This day covers sentiment analysis techniques using Natural Language Processing (NLP) to analyze and classify text data based on emotional tone. Two different approaches are implemented: custom lexicon-based analysis and API-powered news sentiment analysis.

---

## 📂 Project Files

### 1️⃣ `Sentiment_Analysis.ipynb` 🎭

**Custom Lexicon-Based Sentiment Analysis**

#### 🎯 Purpose

Implements a rule-based sentiment classifier using NLTK for text preprocessing and a custom word dictionary approach.

#### 🔧 Key Components

- **Text Preprocessing**: Tokenization, stop word removal, and lemmatization
- **Custom Sentiment Lexicon**: Predefined positive and negative word lists
- **Classification Logic**: Word counting algorithm to determine sentiment

#### 📚 Libraries Used

- `nltk` - Natural Language Toolkit
- `word_tokenize` - Text tokenization
- `stopwords` - Common word filtering
- `WordNetLemmatizer` - Word normalization

#### 🎪 Features

- ✅ Converts text to lowercase
- ✅ Removes stop words
- ✅ Lemmatizes words to base form
- ✅ Counts positive vs negative words
- ✅ Classifies as: positive, negative, or neutral

#### 📊 Sample Output

Analyzes customer reviews and general statements to classify sentiment.

---

### 2️⃣ `News_Sentiment_Analysis.ipynb` 📰

**Real-Time News Sentiment Analysis with API**

#### 🎯 Purpose

Fetches live news headlines from NewsAPI and performs sentiment analysis using TextBlob, with visual representation of results.

#### 🔧 Key Components

- **API Integration**: NewsAPI for fetching US top headlines
- **Sentiment Analysis**: TextBlob polarity scoring
- **Data Visualization**: Horizontal bar chart of sentiment scores

#### 📚 Libraries Used

- `requests` - HTTP requests for API calls
- `pandas` - Data manipulation and analysis
- `textblob` - Automated sentiment analysis
- `matplotlib` - Data visualization

#### 🎪 Features

- 🌐 Fetches real-time news articles
- 📊 Creates DataFrame with titles and descriptions
- 🔍 Handles missing data
- 💯 Calculates sentiment polarity (-1 to +1)
- 📈 Visualizes top 10 headlines with sentiment scores

#### 📊 Visualization

- **X-axis**: Sentiment Score (negative ← 0 → positive)
- **Y-axis**: News Headlines
- **Color**: Sky blue bars for easy reading

---

## 🔑 Key Concepts Learned

### 🧠 NLP Preprocessing

- Tokenization
- Stop word removal
- Lemmatization
- Text normalization

### 📐 Sentiment Analysis Methods

1. **Lexicon-Based**: Manual word lists with counting logic
2. **Machine Learning-Based**: TextBlob's pre-trained model

### 📊 Sentiment Scoring

- **Positive**: > 0 (optimistic, happy, favorable)
- **Neutral**: = 0 (factual, balanced)
- **Negative**: < 0 (pessimistic, critical, unfavorable)

---

## 💡 Use Cases

- 🛒 Customer review analysis
- 📱 Social media monitoring
- 📰 News tone classification
- 💼 Brand reputation tracking
- 🎬 Movie/product feedback analysis

---

## 🎓 Learning Outcomes

✅ Understand text preprocessing pipeline  
✅ Implement custom sentiment classifiers  
✅ Work with external APIs for data collection  
✅ Use pre-trained NLP models (TextBlob)  
✅ Visualize sentiment data effectively  
✅ Compare rule-based vs ML-based approaches

---

## 🧩 Quick Code Snippets (Easy to Memorize!)

### 🔹 Custom Sentiment Analysis Pattern

```python
# 1️⃣ Import & Download
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 2️⃣ Define Word Lists
positive_words = ["love", "great", "amazing", "happy", "good"]
negative_words = ["worst", "terrible", "bad", "awful", "poor"]

# 3️⃣ Preprocess Function
def preprocess(text):
    text = text.lower()  # lowercase
    words = word_tokenize(text)  # tokenize
    words = [w for w in words if w not in stopwords.words('english')]  # remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]  # lemmatize
    return words

# 4️⃣ Analyze Sentiment
def analyze_sentiment(text):
    words = preprocess(text)
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)

    if pos > neg: return "positive"
    elif neg > pos: return "negative"
    else: return "neutral"
```

### 🔹 News API Sentiment Pattern

```python
# 1️⃣ Fetch News
import requests
import pandas as pd
from textblob import TextBlob

API_KEY = "your_api_key"
url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
response = requests.get(url)
articles = response.json()['articles']

# 2️⃣ Create DataFrame
df = pd.DataFrame({
    'Title': [a['title'] for a in articles],
    'Description': [a['description'] for a in articles]
})

# 3️⃣ Analyze Sentiment
df['Sentiment'] = df['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 4️⃣ Visualize
import matplotlib.pyplot as plt
plt.barh(df['Title'][:10], df['Sentiment'][:10], color='skyblue')
plt.xlabel("Sentiment Score")
plt.axvline(0, color='black')
plt.show()
```

### 🔹 Remember This Pattern! 🧠

**Sentiment Analysis Flow:**

1. **Import** → Download resources
2. **Preprocess** → Clean text (lowercase → tokenize → remove stopwords → lemmatize)
3. **Analyze** → Count positive/negative words OR use TextBlob
4. **Classify** → positive/negative/neutral
5. **Visualize** → (Optional) Show results

**Key Formula:**

```
Sentiment = (Positive Count - Negative Count) / Total Words
```

**TextBlob Shortcut:**

```python
TextBlob("text").sentiment.polarity  # Returns -1 to +1
```
