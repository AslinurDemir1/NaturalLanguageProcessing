import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. VERİ SETİ (Label: 1=Olumlu, 0=Olumsuz) ---
data = [
    ("bu ürün harika çok beğendim", 1),
    ("mükemmel bir film kesinlikle izleyin", 1),
    ("tam bir hayal kırıklığı berbat", 0),
    ("hiç beğenmedim paranıza yazık", 0),
    ("kargo çok hızlı geldi teşekkürler", 1),
    ("bozuk çıktı iade edeceğim", 0),
    ("fiyat performans ürünü gayet iyi", 1),
    ("sakın almayın çok kalitesiz", 0)
]

# --- 2. SÖZLÜK OLUŞTURMA (Mühendislik Kısmı) ---
# Tüm cümlelerde geçen kelimeleri tek bir havuzda toplayalım
word_to_ix = {} # Kelime -> Sayı Haritası
for sent, label in data:
    for word in sent.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2 # Olumlu (1) veya Olumsuz (0)

print(f"Sözlük Boyutu: {VOCAB_SIZE} kelime")
print(f"Örnek Sözlük: {list(word_to_ix.items())[:5]}...")

# --- 3. VEKTÖRLEŞTİRME FONKSİYONU (Bag of Words) ---
def make_bow_vector(sentence, word_to_ix):
    # Önce tüm sözlük kadar 0'lardan oluşan bir vektör yap
    vec = torch.zeros(len(word_to_ix))
    # Cümledeki kelimelerin olduğu yerleri 1 yap (veya sayısını artır)
    for word in sentence.split():
        if word in word_to_ix: # Eğer kelimeyi tanıyorsak
            vec[word_to_ix[word]] += 1
    return vec.view(1, -1) # Boyut ekle: [1, VOCAB_SIZE]

# --- 4. MODEL MİMARİSİ ---
class DuyguAnalizModeli(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super(DuyguAnalizModeli, self).__init__()
        # Girdi: Sözlük boyutu kadar (Örn: 30 nöron)
        # Çıktı: 2 nöron (Olumlu/Olumsuz skoru)
        self.linear = nn.Linear(vocab_size, num_labels)
        self.softmax = nn.LogSoftmax(dim=1) # Olasılığa çevir

    def forward(self, x):
        return self.softmax(self.linear(x))

model = DuyguAnalizModeli(VOCAB_SIZE, NUM_LABELS)

# --- 5. EĞİTİM ---
loss_function = nn.NLLLoss() # Negative Log Likelihood (Sınıflandırma için)
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("\nModel Eğitiliyor...")
for epoch in range(100): # 100 tur dön
    for sentence, label in data:
        # A. Hazırlık
        model.zero_grad()
        bow_vec = make_bow_vector(sentence, word_to_ix) # Cümleyi vektöre çevir
        target = torch.tensor([label], dtype=torch.long) # Hedefi tensör yap

        # B. İleri ve Geri Yayılım
        log_probs = model(bow_vec)       # Tahmin et
        loss = loss_function(log_probs, target) # Hatayı bul
        loss.backward()                  # Türev al
        optimizer.step()                 # Güncelle

print("Eğitim Tamamlandı!")

# --- 6. TEST (GERÇEK DÜNYA) ---
def tahmin_et(test_cumlesi):
    with torch.no_grad():
        bow_vec = make_bow_vector(test_cumlesi, word_to_ix)
        log_probs = model(bow_vec)
        # En yüksek skoru al
        tahmin_index = torch.argmax(log_probs, dim=1).item()
        
        durum = "OLUMLU 😊" if tahmin_index == 1 else "OLUMSUZ 😡"
        print(f"Cümle: '{test_cumlesi}' -> {durum}")

print("\n--- SONUÇLAR ---")
tahmin_et("bu film harika")       # Sözlükte var
tahmin_et("ürün berbat sakın almayın") # Sözlükte var
tahmin_et("kargo hızlı geldi")    # Sözlükte var
# Dikkat: Aşağıdaki kelimelerin bazıları sözlükte yok!
tahmin_et("film fena değildi")