import requests
import os
import time
import sys
from urllib.parse import quote

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    
    # Используем абсолютные пути на основе расположения скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    output_dir = os.path.join(project_root, "knowledge_base", "raw")
    os.makedirs(output_dir, exist_ok=True)

    for title in ARTICLES:
        print(f"📥 Загружаю: {title}")
        html = fetch_article(title)
        if html:
            text = html_to_text(html)
            safe_title = title.replace("/", "_").replace("\\", "_")
            output_path = os.path.join(output_dir, f"{safe_title}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        time.sleep(0.4)

if __name__ == "__main__":
    main()