# Anayasa RAG Sohbet Botu (Streamlit - Yerel) 🇹🇷

1982 Türkiye Anayasası'nın güncel metni üzerine sorular sormak için geliştirilmiş, yerel makinede çalışan, Retrieval-Augmented Generation (RAG) tabanlı bir sohbet botu. Anayasa metnini indeksler, anlamsal olarak en ilgili bölümleri **FAISS** ile bulur ve **Google Gemini** API'sini kullanarak cevaplar üretir. Streamlit ile basit bir web arayüzü sunar.

## Özellikler ✨

* **Basit RAG Akışı:** Metni parçala (`chunk`) → Google API ile vektörleştir (`embed`) → FAISS'te sakla (`store`) → FAISS'ten getir (`retrieve`) → Gemini API ile cevap üret (`generate`).
* **Google Gemini LLM:** Cevap üretimi için `models/gemini-2.5-flash` API'si kullanılır.
* **Google Embedding API:** Yüksek kaliteli vektörler için `models/text-embedding-004` API'si kullanılır (yerel model indirme gerektirmez).
* **Streamlit Arayüzü:** Kullanıcı dostu, sohbet tabanlı bir web arayüzü.
* **Kalıcı FAISS Vektör Deposu:** İlk çalıştırmada oluşturulan vektör indeksi (`faiss_index_anayasa_google_emb/` klasöründe), sonraki çalıştırmalar için diske kaydedilir ve otomatik olarak yüklenir, böylece başlangıç süresi kısalır.

## Teknoloji Yığını 🛠️

* **Arayüz (UI):** Streamlit
* **RAG:** LangChain, FAISS (CPU)
* **LLM:** Google Gemini (`models/gemini-2.5-flash`) - API
* **Embeddings:** Google (`models/text-embedding-004`) - API
* **Veri:** `anayasa.txt` (Yerel metin dosyası)

## Gereksinimler 📋

* Python 3.10+
* Geçerli bir **Google API anahtarı** (Gemini ve Text Embedding API'lerine erişimi olan). Google AI Studio veya Google Cloud Console'dan alınabilir.
* (Önerilir) Git (Kodu GitHub'dan çekmek veya versiyonlamak için).

## Kurulum ⚙️

1.  **Projeyi Edinin:**
    * Eğer GitHub'daysa, repoyu klonlayın:
        ```bash
        git clone [https://github.com/KULLANICI_ADIN/REPO_ADIN.git](https://github.com/KULLANICI_ADIN/REPO_ADIN.git)
        cd REPO_ADIN
        ```
    * Veya proje dosyalarını (`app.py`, `anayasa.txt`) içeren bir klasör oluşturun.

2.  **(Önerilir) Sanal Ortam Oluşturun ve Aktive Edin:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Bağımlılıkları Yükleyin:**
    * Proje klasöründe bir `requirements.txt` dosyası varsa:
        ```bash
        pip install -r requirements.txt
        ```
    * Veya gerekli paketleri doğrudan kurun (minimal liste):
        ```bash
        pip install -U "protobuf>=5.29.1" streamlit langchain langchain-core langchain-community langchain-text-splitters langchain-google-genai google-generativeai faiss-cpu python-dotenv
        ```

4.  **Ortam Değişkenini Yapılandırın:**
    * Proje kök dizininde (`app.py`'nin yanında) **`.env`** adında bir dosya oluşturun.
    * İçine **sadece** şu satırı ekleyin (kendi anahtarınızla değiştirin):
        ```
        GOOGLE_API_KEY=AIzaSy...senin_google_api_anahtarın_buraya...
        ```
    * **(Alternatif)** Terminalde `streamlit run app.py` demeden önce ortam değişkenini manuel olarak da ayarlayabilirsiniz (her yeni terminal oturumu için tekrarlamanız gerekir):
        * Windows (PowerShell): `$env:GOOGLE_API_KEY = "AIzaSy..."`
        * Windows (CMD): `set GOOGLE_API_KEY=AIzaSy...`
        * Mac/Linux: `export GOOGLE_API_KEY=AIzaSy...`

5.  **Vektör Veritabanı Otomatik Oluşturulacak:** Ayrı bir `create_database.py` adımına gerek yoktur. `app.py` ilk kez çalıştığında `faiss_index_anayasa_google_emb/` klasörünü bulamazsa, `anayasa.txt`'yi işleyip Google Embedding API'sine çağrılar yaparak indeksi oluşturacak ve kaydedecektir.
    * ⚠️ **Uyarı:** Bu ilk indeks oluşturma işlemi, anayasa metninin uzunluğuna bağlı olarak **birkaç dakika sürebilir** ve Google API kotanızı kullanır. Sonraki çalıştırmalar çok daha hızlı olacaktır.

6.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```

7.  Tarayıcınızı açın ve genellikle `http://localhost:8501` adresine gidin.

## Yapılandırma (Varsayılanlar - `app.py` içinde) ⚙️

* **Üretici Model (LLM):** `models/gemini-2.5-flash` (Sıcaklık `temperature=0`)
* **Embedding Modeli:** `models/text-embedding-004` (Google API)
* **Parçalama (Chunking):** 1000 karakter boyut, 200 karakter örtüşme (`overlap`)
* **Getirme (Retrieval) k:** En benzer 5 parça (`k=5`)

## Proje Yapısı 📁
