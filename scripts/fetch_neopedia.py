import requests
import os
import time
from urllib.parse import quote

API_URL = "https://neopedia.fandom.com/ru/api.php"

def fetch_article(title):
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": 1
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        data = response.json()
        if "error" in data:
            print(f"❌ Ошибка при загрузке '{title}': {data['error']['info']}")
            return None
        html = data["parse"]["text"]["*"]
        return html
    except Exception as e:
        print(f"⚠️ Исключение при загрузке '{title}': {e}")
        return None

def html_to_text(html):
    import re
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"\[\d+\]|\[править\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    from articles_list import ARTICLES
    os.makedirs("../knowledge_base/raw", exist_ok=True)

    for title in ARTICLES:
        print(f"📥 Загружаю: {title}")
        html = fetch_article(title)
        if html:
            text = html_to_text(html)
            safe_title = title.replace("/", "_").replace("\\", "_")
            with open(f"../knowledge_base/raw/{safe_title}.txt", "w", encoding="utf-8") as f:
                f.write(text)
        time.sleep(0.4)

if __name__ == "__main__":
    main()