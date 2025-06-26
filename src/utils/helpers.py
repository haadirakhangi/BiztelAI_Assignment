# src/utils/helpers.py
import re
from urllib.parse import urlparse

def clean_article_url_to_text(url: str) -> str:
    """
    Cleans a Washington Post article URL to extract meaningful keywords for searching.
    Focuses on the slug (human-readable title part) of the URL.
    Example: "https://www.washingtonpost.com/sports/colleges/good-night-keith-jackson-and-thanks/2018/01/13/79aff714-f898-11e7-b34a-b85626af34ef_story.html"
    Becomes: "sports colleges good night keith jackson and thanks" (potentially including date parts if they are part of slug or context)
    """
    if not url:
        return ""
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        
        # Remove the UUID and _story.html suffix
        path = re.sub(r'/[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}_story\.html$', '', path)
        path = re.sub(r'_story\.html$', '', path) 
        path = re.sub(r'\.html$', '', path) 
        parts = path.split('/')
        descriptive_parts = []
        for part in parts:
            if part.lower() in ["www", "washingtonpost", "com", "wp-dyn", "content", "pb", "video", "posttv", "glogin", "gdpr-consent", "local", "national", "world", "business", "technology", "sports", "opinions", "entertainment", "lifestyle", "politics"]:
                descriptive_parts.append(part.replace('-', ' '))
            elif re.match(r'^\d{4}$', part) or re.match(r'^\d{1,2}$', part): 
                descriptive_parts.append(part)
            else:
                descriptive_parts.append(part.replace('-', ' '))
        
        cleaned_text = " ".join(descriptive_parts)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text.lower()
        
    except Exception as e:
        print(f"Error cleaning URL {url}: {e}")
        name_part = url.split('/')[-1].replace('_story.html', '').replace('.html', '')
        name_part = name_part.split('_')[0] 
        return name_part.replace('-', ' ').lower()