import streamlit as st
from transformers import pipeline

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Benim YZ Yazarım", page_icon="🤖")

st.title("🤖 Türkçe Yapay Zeka Yazarı")
st.write("Başlangıç cümlesini yazın, gerisini yapay zeka tamamlasın!")

# --- MODELİ YÜKLEME (ÖNBELLEKLİ) ---
# @st.cache_resource sayesinde modeli her seferinde tekrar indirmez, hafızada tutar.
@st.cache_resource
def model_yukle():
    return pipeline('text-generation', model='redrussianarmy/gpt2-turkish-cased')

# Yükleniyor mesajı gösterelim
with st.spinner('Yapay Zeka Beyni Yükleniyor... Lütfen bekleyin...'):
    generator = model_yukle()

# --- KULLANICI ARAYÜZÜ ---
# Kullanıcıdan metin alma kutusu
baslangic_metni = st.text_input("Cümlenin başını yazın:", "Yapay zeka gelecekte")

# Ayarlar çubuğu (Sidebar)
st.sidebar.header("Yaratıcılık Ayarları")
uzunluk = st.sidebar.slider("Maksimum Kelime Sayısı", min_value=10, max_value=200, value=100)
yaraticilik = st.sidebar.slider("Yaratıcılık (Temperature)", min_value=0.1, max_value=1.5, value=0.9)

# --- BUTON VE SONUÇ ---
if st.button("Yazıyı Tamamla"):
    with st.spinner('Yazıyorum...'):
        try:
            sonuclar = generator(
                baslangic_metni, 
                max_length=uzunluk, 
                num_return_sequences=1, 
                temperature=yaraticilik,
                repetition_penalty=1.2 # Tekrar etmeyi engellemek için ceza ekledik!
            )
            
            uretilen_yazi = sonuclar[0]['generated_text']
            
            # Sonucu güzel bir kutuda göster
            st.success("İşte Yapay Zekanın Devamı:")
            st.text_area("", value=uretilen_yazi, height=200)
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

st.markdown("---")
st.caption("Bu proje Python, Transformers ve Streamlit kullanılarak yapılmıştır.")