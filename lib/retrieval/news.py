import requests
import os
import json
from bs4 import BeautifulSoup, SoupStrainer
from markdownify import markdownify as md
import yfinance as yf

def get_news_gnews(query: str):
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

def get_finnews_seekingalpha(query: str):
    newsurl = f"https://seekingalpha.com/api/v3/searches/news?filter[query]={'%20'.join(query.split(' '))}"

    headers = {
        "User-Agent" : "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language" : "en-GB,en;q=0.5",
        "Priority" : "u=0, i",
    }

    newsres = requests.get(newsurl, headers=headers)
    newsres.raise_for_status()
    newshtml = newsres.text
    newscontent = json.loads(newshtml)

    links = []
    for elem in newscontent["data"]:
        links.append("https://seekingalpha.com" + elem["url"])
    links.sort()
    links.reverse()

    cleaned_articles = []
    for item in links[:1]:
        try:
            headers = {
                "User-Agent" : "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language" : "en-GB,en;q=0.5",
                "Priority" : "u=0, i",
                "Cookie" : 'machine_cookie=fzyzrlo73c1758296129; OptanonConsent=isGpcEnabled=1&datestamp=Fri+Sep+26+2025+11%3A05%3A53+GMT%2B0100+(British+Summer+Time)&version=202411.1.0&browserGpcFlag=1&isIABGlobal=false&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A0%2CC0003%3A0%2CC0004%3A0%2CC0005%3A0&hosts=H74%3A1%2CH78%3A1%2CH17%3A1%2CH72%3A0%2CH95%3A0%2CH76%3A0%2CH79%3A0%2CH19%3A0%2CH30%3A0%2CH93%3A0%2CH85%3A0%2CH86%3A0%2CH94%3A0%2CH87%3A0%2CH89%3A0%2CH90%3A0%2CH81%3A0%2CH16%3A0%2CH82%3A0&genVendors=V12%3A0%2CV16%3A1%2CV7%3A0%2CV8%3A0%2CV13%3A1%2CV3%3A1%2CV6%3A0%2CV14%3A0%2CV17%3A0%2CV1%3A1%2CV4%3A1%2CV9%3A0%2C&geolocation=GB%3BENG&AwaitingReconsent=false; g_state={"i_p":1758303685041,"i_l":1}; user_id=58405064; user_nick=; user_devices=; user_cookie_key=i3r2ff; u_voc=; sapu=101; user_remember_token=440f9f3ca42d21aa396dd8a598409447096858b8; gk_user_access=1**1758874424; gk_user_access_sign=9c1eb8a34192c8aba1b59693c7d0a6a957e4d4c1; OptanonAlertBoxClosed=2025-09-19T15:42:56.004Z; _pxvid=3709bddd-97bf-11f0-8d17-7bf197c36a43; LAST_VISITED_PAGE=%7B%22pathname%22%3A%22https%3A%2F%2Fseekingalpha.com%2Fnews%2F4483901-sp500-nasdaq-composite-dow-jones-stock-market-news-rate-cuts%22%2C%22fromMpArticle%22%3Afalse%7D; user_locale=en; session_id=ad96c79d-cf9a-41b1-84f3-8bfd7af91cf0; _sasource=; userLocalData_mone_session_lastSession=%7B%22machineCookie%22%3A%22fzyzrlo73c1758296129%22%2C%22machineCookieSessionId%22%3A%22fzyzrlo73c1758296129%261758878928519%22%2C%22sessionStart%22%3A1758878928519%2C%22sessionEnd%22%3A1758882952463%2C%22firstSessionPageKey%22%3A%22e597cd75-8ecd-473e-93d6-15dae0f7b826%22%2C%22isSessionStart%22%3Afalse%2C%22lastEvent%22%3A%7B%22event_type%22%3A%22mousemove%22%2C%22timestamp%22%3A1758881152463%7D%7D',
                "Connection" : "keep-alive",
            }
            response = requests.get(item, headers=headers, timeout=3)
            response.raise_for_status()
            html = response.text
            cleaned_articles.append(html)
            continue

            soup = BeautifulSoup(html, 'html.parser')

            article_content = ""
            for s in soup(['script', 'style', 'header', 'footer', 'nav']):
                s.decompose()
            
            body_html = soup.body
            if body_html:
                article_content = md(str(body_html), heading_style="ATX").strip()

            if article_content:
                cleaned_articles.append(article_content)

        except Exception as e:
            print(e)
            continue
            
    return cleaned_articles

def get_finnews_yfinance(ticker: str):
    news_items = yf.Search(ticker, news_count=5).news
    cleaned_articles = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for item in news_items:
        link = item.get("link")
        title = item.get("title")
        if not link or not title:
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
                cleaned_articles.append({"title": title, "content": article_content})

        except Exception as e:
            continue
            
    return cleaned_articles

with open("./seekingalpha.html", "w") as f:
    f.write(get_finnews_seekingalpha("fed rates")[0])