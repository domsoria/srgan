from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import os
from tqdm import tqdm

def add_noise(image):
    """Aggiungi rumore casuale all'immagine."""
    np_image = np.array(image)
    noise = np.random.normal(0, 25, np_image.shape).astype(np.uint8)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Assicura che i valori siano nell'intervallo corretto
    return Image.fromarray(noisy_image)

def degrade_image(image_path, output_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Converti l'immagine in modalità RGB
    
    # Applica sfocatura gaussiana
    degraded_image = image.filter(ImageFilter.GaussianBlur(2))
    
    # Aggiungi rumore
    degraded_image = add_noise(degraded_image)
    
    # Regola la luminosità e il contrasto
    enhancer = ImageEnhance.Contrast(degraded_image)
    degraded_image = enhancer.enhance(0.7)  # Riduce il contrasto
    enhancer = ImageEnhance.Brightness(degraded_image)
    degraded_image = enhancer.enhance(1.2)  # Aumenta leggermente la luminosità
    
    # Aggiungi graffi e polvere
    np_image = np.array(degraded_image)
    for _ in range(100):  # Aggiungi 100 punti neri
        x, y = np.random.randint(0, np_image.shape[1]), np.random.randint(0, np_image.shape[0])
        np_image[y, x] = 0
    for _ in range(100):  # Aggiungi 100 punti bianchi
        x, y = np.random.randint(0, np_image.shape[1]), np.random.randint(0, np_image.shape[0])
        np_image[y, x] = 255
    degraded_image = Image.fromarray(np_image)
    
    degraded_image.save(output_path, 'JPEG')  # Salva l'immagine degradata in formato JPEG

high_quality_dir = 'dataset/high_quality/'
low_quality_dir = 'dataset/low_quality/'

if not os.path.exists(low_quality_dir):
    os.makedirs(low_quality_dir)

image_files = os.listdir(high_quality_dir)

# Utilizza tqdm per creare una barra di avanzamento
for filename in tqdm(image_files, desc="Degradazione delle immagini", unit="immagine"):
    degrade_image(os.path.join(high_quality_dir, filename), os.path.join(low_quality_dir, filename))

print("Processo di degradazione completato.")
