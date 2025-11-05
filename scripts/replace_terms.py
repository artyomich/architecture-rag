import os
import json
import re

def load_replacements():
    with open("terms_map.json", "r", encoding="utf-8") as f:
        return json.load(f)

def apply_replacements(text, replacements):
    sorted_terms = sorted(replacements.keys(), key=len, reverse=True)
    for term in sorted_terms:
        pattern = re.escape(term)
        text = re.sub(pattern, replacements[term], text, flags=re.IGNORECASE)
    return text

def main():
    replacements = load_replacements()
    os.makedirs("../knowledge_base/final", exist_ok=True)

    for filename in os.listdir("../knowledge_base/raw"):
        if filename.endswith(".txt"):
            with open(f"../knowledge_base/raw/{filename}", "r", encoding="utf-8") as f:
                text = f.read()
            new_text = apply_replacements(text, replacements)
            with open(f"../knowledge_base/final/{filename}", "w", encoding="utf-8") as f:
                f.write(new_text)
            print(f"✅ Обработано: {filename}")

if __name__ == "__main__":
    main()