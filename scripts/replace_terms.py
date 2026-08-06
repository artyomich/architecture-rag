import os
import json
import re
import sys

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_replacements():
    """Загружает карту терминов из JSON файла."""
    terms_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terms_map.json")
    with open(terms_map_path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_replacements(text, replacements):
    sorted_terms = sorted(replacements.keys(), key=len, reverse=True)
    for term in sorted_terms:
        pattern = re.escape(term)
        text = re.sub(pattern, replacements[term], text, flags=re.IGNORECASE)
    return text

def main():
    replacements = load_replacements()
    
    # Используем абсолютные пути на основе расположения скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    input_dir = os.path.join(project_root, "knowledge_base", "raw")
    output_dir = os.path.join(project_root, "knowledge_base", "final")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            new_text = apply_replacements(text, replacements)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(new_text)
            print(f"✅ Обработано: {filename}")

if __name__ == "__main__":
    main()