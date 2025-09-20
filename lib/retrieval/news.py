import requests
import os
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def get_gnews(query: str):
    aggresurl = f"https://gnews.io/api/v4/search?q={'+'.join(query.split(' '))}&apikey={os.environ['GNEWS_TOKEN']}"
    aggres = requests.get(aggresurl)
    aggres.raise_for_status()

    news_items = json.loads(aggres.text)["articles"]
    cleaned_articles = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for item in news_items:
        link = item.get("url")
        title = item.get("title")
        date = item.get("publishedAt")
        if not link or not title or not date:
            continue

        try:
            response = requests.get(link, headers=headers, timeout=3)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            article_content = ""
            for s in soup(['script', 'style', 'header', 'footer', 'nav']):
                s.decompose()
            
            body_html = soup.body
            if body_html:
                article_content = md(str(body_html), heading_style="ATX").strip()

            if article_content:
                cleaned_articles.append({"title": title, "content": article_content, "date": date})

        except Exception as e:
            continue
            
    return cleaned_articles

print(get_gnews("Nepal corruption"))