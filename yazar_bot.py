import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. VERİ HAZIRLIĞI (ALFABEYİ ÖĞRETMEK) ---
cumle = "yapay zeka öğrenmek çok eğlenceli."

# Bilgisayar harfleri bilmez, sayıları bilir. Sözlük yapıyoruz.
# set() ile benzersiz harfleri bulduk, list() ile sıraladık.
karakterler = sorted(list(set(cumle))) 

# Harf -> Sayı (Encoder)
char_to_int = {ch: i for i, ch in enumerate(karakterler)}
# Sayı -> Harf (Decoder)
int_to_char = {i: ch for i, ch in enumerate(karakterler)}

print(f"Öğrenilecek Cümle: '{cumle}'")
print(f"Karakter Kümesi: {karakterler}")

# Veriyi sayıya çevirelim (Tensor formatı)
# Cümleyi sayı dizisine dönüştürdük
encoded_cumle = [char_to_int[ch] for ch in cumle]
# PyTorch tensörüne çevir ve boyut ekle (Batch size = 1)
input_seq = torch.tensor(encoded_cumle).unsqueeze(0)  # Boyut: [1, KarakterSayisi]

# --- 2. MODEL MİMARİSİ (RNN) ---
class YazarBot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(YazarBot, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding: Sayıları (örn: 5) vektöre (örn: [0.1, -0.5...]) çevirir.
        # Bu, harfler arasındaki ilişkiyi anlamasını kolaylaştırır.
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # RNN Katmanı: Hafızayı tutan yer
        # batch_first=True: Girdi formatı [Batch, Uzunluk, Özellik] olsun diye.
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        
        # Çıktı Katmanı: Bir sonraki harfin hangisi olacağını tahmin eder.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: Giriş harfleri (Sayılar)
        # hidden: Önceki harften gelen hafıza
        
        x = self.embedding(x) # Sayıları vektöre çevir
        
        # out: Modelin çıkardığı özellikler
        # hidden: Bir sonraki adıma aktarılacak hafıza
        out, hidden = self.rnn(x, hidden)
        
        # Sonuçları harf tahminine dönüştür
        out = out.reshape(-1, self.hidden_size) # Düzleştir
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        # Hafızayı sıfırla (İlk başlangıç)
        return torch.zeros(1, batch_size, self.hidden_size)

# --- 3. AYARLAR ---
input_size = len(karakterler) # Alfabemizdeki harf sayısı
hidden_size = 32              # Hafıza kapasitesi
output_size = len(karakterler) # Çıktı da yine alfabeden bir harf olacak

model = YazarBot(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimizer genelde NLP'de daha iyidir

# --- 4. EĞİTİM DÖNGÜSÜ ---
print("\nYazar Bot Eğitiliyor...")

for epoch in range(500): # 500 kere aynı cümleyi okusun
    optimizer.zero_grad()
    
    # Girdi: "yapay zeka öğrenmek çok eğlenceli" (Son nokta hariç)
    # Hedef: "apay zeka öğrenmek çok eğlenceli." (İlk harf hariç, bir kaydırılmış)
    input_data = input_seq[:, :-1] 
    target_data = input_seq[:, 1:]
    
    hidden = model.init_hidden(1) # Hafızayı temizle
    
    output, hidden = model(input_data, hidden)
    
    # Boyutları düzelt ve hatayı hesapla
    loss = criterion(output, target_data.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Tur {epoch+1}, Hata: {loss.item():.4f}')

# --- 5. TEST (METİN ÜRETİMİ) ---
print("\n--- Hadi Yazdıralım! ---")

def cumle_tamamla(baslangic_harfi, uzunluk=30):
    model.eval()
    
    # Başlangıç harfini hazırla
    input_char = torch.tensor([[char_to_int[baslangic_harfi]]])
    hidden = model.init_hidden(1)
    
    tahmin_edilen_cumle = baslangic_harfi
    
    with torch.no_grad():
        for _ in range(uzunluk):
            output, hidden = model(input_char, hidden)
            
            # En yüksek ihtimalli harfi seç
            top_i = output.argmax(1)
            char_index = top_i.item()
            
            # Harfe çevir ve cümleye ekle
            harf = int_to_char[char_index]
            tahmin_edilen_cumle += harf
            
            # Bu harfi bir sonraki adımın girdisi yap (Oto-Regresif)
            input_char = torch.tensor([[char_index]])
            
    return tahmin_edilen_cumle

print(f"Giriş 'y': {cumle_tamamla('y')}")
print(f"Giriş 'z': {cumle_tamamla('z')}")
print(f"Giriş 'ö': {cumle_tamamla('ö')}")