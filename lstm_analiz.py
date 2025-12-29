import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. VERİ HAZIRLIĞI ---
# Veri setimiz (Cümle, Etiket)
# 1: Olumlu, 0: Olumsuz
train_data = [
    ("bu film gerçekten harika ve sürükleyici", 1),
    ("oyunculuklar berbat senaryo çok kötü", 0),
    ("hayatımda izlediğim en iyi film", 1),
    ("zaman kaybı sakın izlemeyin", 0),
    ("kurgu muazzam efektler çok başarılı", 1),
    ("hiç beğenmedim çok sıkıcıydı", 0),
    ("sonu çok saçma bitti", 0),
    ("mutlaka izlenmesi gereken bir başyapıt", 1)
]

# Sözlük Oluşturma (Kelimeleri Sayıya Çevirme)
word_to_ix = {"<PAD>": 0} # Dolgu (Padding) için özel karakter
for sent, label in train_data:
    for word in sent.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(f"Sözlük Boyutu: {len(word_to_ix)}")

# --- 2. MÜHENDİSLİK KISMI: PADDING (DOLGU) ---
# LSTM, girdilerin hepsinin aynı boyda olmasını ister.
# Kısa cümlelerin sonuna 0 (PAD) ekleyerek hepsini en uzun cümleye eşitleyeceğiz.
MAX_LEN = 6  # Cümleleri 6 kelimeye sabitleyelim

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq.split():
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0) # Bilinmeyen kelime (veya PAD)
            
    # Eğer 6'dan kısaysa sonuna 0 ekle
    if len(idxs) < MAX_LEN:
        idxs += [0] * (MAX_LEN - len(idxs))
    # Eğer 6'dan uzunsa kes
    else:
        idxs = idxs[:MAX_LEN]
        
    return torch.tensor(idxs, dtype=torch.long).view(1, -1) # [1, 6] boyutunda

# --- 3. LSTM MODEL MİMARİSİ ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        
        # 1. Embedding Katmanı: Kelime ID -> Vektör
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Katmanı: Vektör Dizisi -> Hafıza Özeti
        # batch_first=True: Girdi [Batch, Uzunluk, Özellik] formatında olsun
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # 3. Çıktı Katmanı: Hafıza -> Karar (0/1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [1, 6] (Kelimelerin ID'leri)
        
        embeds = self.embedding(x) 
        # embeds: [1, 6, 10] (Her kelime 10'luk vektör oldu)
        
        # LSTM Çalışıyor...
        # out: Tüm adımların çıktısı
        # hidden: Kısa vadeli hafıza (Son durum)
        # cell: Uzun vadeli hafıza (Hücre durumu)
        lstm_out, (hidden, cell) = self.lstm(embeds)
        
        # Bize sadece son adımın hafızası lazım (Cümlenin özeti)
        # hidden[0]: [1, 1, 16] -> [1, 16]
        final_hidden = hidden[-1] 
        
        # Karar ver
        prediction = self.fc(final_hidden)
        return self.sigmoid(prediction)

# --- 4. MODEL AYARLARI ---
EMBEDDING_DIM = 10  # Her kelime 10 sayılık bir vektör olsun
HIDDEN_DIM = 16     # LSTM'in hafızasında 16 özellik tutulsun
OUTPUT_DIM = 1      # Çıktı tek bir sayı (0-1 arası olasılık)

model = LSTMClassifier(len(word_to_ix), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
loss_function = nn.BCELoss() # Binary Cross Entropy (İkili Sınıflandırma)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 5. EĞİTİM ---
print("\nLSTM Eğitiliyor...")
for epoch in range(300): # 300 tur
    total_loss = 0
    for sentence, label in train_data:
        model.zero_grad()
        
        # Veriyi hazırla
        inputs = prepare_sequence(sentence, word_to_ix)
        target = torch.tensor([[float(label)]]) # Hedef: [1.0] veya [0.0]
        
        # İleri ve Geri
        y_pred = model(inputs)
        loss = loss_function(y_pred, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch+1) % 50 == 0:
        print(f"Tur {epoch+1}, Hata: {total_loss:.4f}")

# --- 6. TEST ZAMANI ---
print("\n--- Gerçek Test ---")

def test_et(cumle):
    model.eval() # Değerlendirme modu
    with torch.no_grad():
        inputs = prepare_sequence(cumle, word_to_ix)
        score = model(inputs).item()
        
        durum = "OLUMLU 😄" if score > 0.5 else "OLUMSUZ 😡"
        print(f"Cümle: '{cumle}'")
        print(f"Skor: %{score*100:.2f} -> {durum}")
        print("-" * 30)

test_et("bu film harika")
test_et("senaryo çok sıkıcıydı")
test_et("efektler başarılı")
test_et("zaman kaybı")
# Görmediği bir cümle deneyelim
test_et("oyunculuklar çok kötü")