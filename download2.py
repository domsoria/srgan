import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def is_valid_image(response):
    try:
        img = Image.open(BytesIO(response.content))
        img.verify()  # Verifica che sia un'immagine valida
        return True
    except Exception as e:
        return False

def download_images(query, num_images):
    ddgs = DDGS()
    results = ddgs.images(query, max_results=num_images)
    
    query_dir = os.path.join("dataset", query)
    os.makedirs(query_dir, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.google.com/'  # Modifica il referer se necessario
    }

    for i, result in enumerate(tqdm(results, desc=f"Scaricando immagini per '{query}'", unit="immagine")):
        image_url = result['image']
        try:
            img_response = requests.get(image_url, headers=headers, timeout=10)  # Timeout di 10 secondi
            img_response.raise_for_status()
            if is_valid_image(img_response):
                img = Image.open(BytesIO(img_response.content))
                img_format = img.format.lower() if img.format else 'jpg'
                img.save(os.path.join(query_dir, f"{query}_{i}.{img_format}"))
            else:
                print(f"Errore: il file scaricato non è un'immagine valida: {image_url}")
        except requests.exceptions.Timeout:
            print(f"Errore: Timeout superato per l'URL: {image_url}")
        except Exception as e:
            print(f"Errore nel download dell'immagine {image_url}: {e}")

# Definisci le parole chiave e il numero di immagini da scaricare
search_terms = ["cats", "dogs", "landscape", "house", "city", "car", "animals", "people", "men", "women"]
num_images = 200

# Scarica le immagini per ciascun termine di ricerca
for term in search_terms:
    print(f"Inizio download per '{term}'...")
    download_images(term, num_images)
    print(f"Completato per '{term}'.")

print("Download del dataset completato.")
