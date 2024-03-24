from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
from bs4 import BeautifulSoup

# Step 1: Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize the NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Step 2: Define a function to fetch and parse content from a URL
def fetch_news_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # This is a simplified example; you might need to adjust selectors based on the site's structure
        text_content = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text_content
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

# Step 3: Define a function to extract entities using the NER model
def extract_entities(text):
    entities = ner_pipeline(text)
    return entities

# Example URLs - You would replace these with actual article URLs from CNN, WSJ, NY Times
example_urls = [
    "https://www.cnn.com/2023/03/24/health/example-article/index.html",
    "https://www.wsj.com/articles/example-article",
    "https://www.nytimes.com/2023/03/24/health/example-article.html"
]

# Fetch, process, and display entities for each URL
for url in example_urls:
    print(f"Processing {url}")
    content = fetch_news_content(url)
    if content:
        entities = extract_entities(content)
        print(entities)
    else:
        print("No content found or error occurred.")
