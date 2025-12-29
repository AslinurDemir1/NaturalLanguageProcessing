import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- VERİ SETİ ---
# Bilgisayara bu cümleleri vereceğiz
cumleler = [
    "Bu sabah elma ve armut yedim.",      # 0. Cümle (Meyve)
    "Pazardan taze meyve aldım.",         # 1. Cümle (Meyve - Kelime benzerliği az ama anlam yakın)
    "Yeni aldığım araba çok hızlı.",      # 2. Cümle (Alakasız - Taşıt)
    "Kırmızı spor arabaları severim."     # 3. Cümle (Taşıt)
]

# --- 1. VEKTÖRLEŞTİRME (TF-IDF) ---
# TF-IDF: Bir kelime bir cümlede çok geçiyor ama tüm belgelerde az geçiyorsa değerlidir.
# Bu yöntem kelimeleri sayısal vektörlere çevirir.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cumleler)

# Vektörlerin boyutunu görelim (Kaç cümle x Kaç kelime)
print(f"Vektör Matrisi Boyutu: {tfidf_matrix.shape}")

# --- 2. BENZERLİK HESAPLAMA (COSINE SIMILARITY) ---
# İlk cümle ("...elma armut yedim") ile diğerlerinin benzerliğine bakalım.
hedef_cumle_index = 0 

print(f"\nHEDEF CÜMLE: '{cumleler[hedef_cumle_index]}'\n")

similarities = cosine_similarity(tfidf_matrix[hedef_cumle_index], tfidf_matrix)

# Sonuçları yazdıralım
for i in range(len(cumleler)):
    if i == hedef_cumle_index: continue # Kendisiyle kıyaslamayı geç
    
    skor = similarities[0][i]
    print(f"Kıyaslanan: '{cumleler[i]}'")
    print(f"Benzerlik Skoru: %{skor*100:.2f}")
    print("-" * 30)