import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. VERİ HAZIRLIĞI ---
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

# Sözlük oluşturma (Kelime -> Index)
word_to_ix = {}
for sent, label in data:
    for word in sent.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

# Cümleyi index listesine çeviren fonksiyon
def make_index_vector(sentence, word_to_ix):
    idxs = []
    for word in sentence.split():
        # Eğer kelime sözlükte varsa indexini al, yoksa görmezden gel (veya <UNK> yap)
        if word in word_to_ix:
            idxs.append(word_to_ix[word])
    return torch.tensor(idxs, dtype=torch.long)

# --- 2. MODEL MİMARİSİ (EMBEDDING İLE) ---
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super(EmbeddingModel, self).__init__()
        
        # 1. KATMAN: EMBEDDING
        # Her kelimeyi 'embed_dim' boyutunda (örn: 10) bir vektöre çevirir.
        # Bu vektörler eğitim sırasında GÜNCELLENİR ve ANLAM kazanır.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. KATMAN: KARAR
        # Embedding çıktısını alıp sonuca götürür.
        self.linear = nn.Linear(embed_dim, num_labels)

    def forward(self, inputs):
        # inputs: [3, 12, 5...] gibi kelime indexleri
        
        embeds = self.embedding(inputs) 
        # embeds şu an şöyle: [[0.1, -0.5..], [0.9, 0.1..]...] (Kelime Sayısı x Embed Boyutu)
        
        # Mühendislik Sorunu: Cümle uzunluğu değişken ama Linear katman sabit girdi ister.
        # Çözüm: Tüm kelime vektörlerinin ORTALAMASINI veya TOPLAMINI alarak tek bir özet vektör çıkarırız.
        # Burada basitçe toplamını alıyoruz (Sum Pooling).
        bow_vector = torch.sum(embeds, dim=0) 
        
        # Sınıflandırma
        out = self.linear(bow_vector.view(1, -1))
        return torch.log_softmax(out, dim=1)

# Ayarlar: Her kelimeyi 10 sayılık bir vektörle temsil et
EMBED_DIM = 10 
model = EmbeddingModel(VOCAB_SIZE, EMBED_DIM, NUM_LABELS)

# --- 3. EĞİTİM ---
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Embedding Modeli Eğitiliyor...")
for epoch in range(100):
    total_loss = 0
    for sentence, label in data:
        model.zero_grad()
        
        # Veriyi hazırla
        input_vec = make_index_vector(sentence, word_to_ix)
        target = torch.tensor([label], dtype=torch.long)
        
        # Eğitim adımı
        log_probs = model(input_vec)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

print(f"Eğitim Bitti. Son Hata: {total_loss:.4f}")

# --- 4. TEST VE VEKTÖR İNCELEME ---
print("\n--- Test Sonuçları ---")

def test_et(cumle):
    with torch.no_grad():
        vec = make_index_vector(cumle, word_to_ix)
        if len(vec) == 0: return # Boş cümle kontrolü
        probs = model(vec)
        tahmin = torch.argmax(probs).item()
        etiket = "OLUMLU" if tahmin == 1 else "OLUMSUZ"
        print(f"'{cumle}' -> {etiket}")

test_et("film harika")
test_et("ürün çok kalitesiz")

# --- 5. MÜHENDİSLİK VİZYONU: KELİME VEKTÖRLERİNİ GÖRMEK ---
# Modelin "harika" ve "berbat" kelimeleri için öğrendiği sayıları görelim.
print("\n--- Modelin Kelimelere Verdiği Matematiksel Değerler ---")
kelimeler = ["harika", "berbat"]
for k in kelimeler:
    idx = word_to_ix[k]
    vektor = model.embedding.weight[idx]
    print(f"'{k}' Vektörü:\n {vektor.detach().numpy().round(2)}")