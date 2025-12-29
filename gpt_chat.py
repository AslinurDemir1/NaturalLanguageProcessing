from transformers import pipeline, set_seed

# --- 1. MODELİ ÇAĞIRMA ---
# Hugging Face'in "pipeline" özelliği, karmaşık kodları tek satıra indirir.
# 'text-generation': Metin üretmek istiyorum diyoruz.
# 'model': Kullanacağımız hazır beyin (Topluluk tarafından eğitilmiş Türkçe GPT-2)
print("Model indiriliyor ve hazırlanıyor... (İlk seferde uzun sürebilir)")

try:
    # Türkçe için eğitilmiş popüler bir GPT-2 modelini kullanıyoruz
    generator = pipeline('text-generation', model='redrussianarmy/gpt2-turkish-cased')
except Exception as e:
    print(f"Hata oluştu: {e}")
    print("Lütfen 'pip install torch transformers' komutunu çalıştırdığından emin ol.")
    exit()

# --- 2. AYARLAR (YARATICILIK DÜĞMELERİ) ---
# set_seed(42) # Eğer bunu açarsan her zaman aynı cevabı verir (Bilimsel deney için)

def yazi_yaz(baslangic_cumlesi):
    print(f"\n📝 Giriş: '{baslangic_cumlesi}'")
    print("-" * 30)
    
    # Model çalışıyor...
    # max_length: En fazla kaç kelime/token yazsın?
    # num_return_sequences: Kaç farklı varyasyon yazsın?
    # temperature: Yaratıcılık ayarı. (0.7 dengeli, 1.5 çılgın, 0.1 robotik)
    cikti = generator(baslangic_cumlesi, max_length=100, num_return_sequences=1, temperature=0.9)
    
    # Sonucu temizleyip yazdıralım
    uretilen_metin = cikti[0]['generated_text']
    print(f"🤖 YZ: {uretilen_metin}")
    print("-" * 30)

# --- 3. DENEME ZAMANI ---
yazi_yaz("Yapay zeka mühendisliği okumak")
yazi_yaz("Türkiye'nin en güzel şehri")
yazi_yaz("Bugün hava çok güzel olduğu için")