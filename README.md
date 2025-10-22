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
    projeyi, projenin ana sayfasındaki "code" yazan yeşil kutuya tıklayarak zip olarak bilgisayarınıza indirin ve ardından PyCharm vb python destekleyen programınızda açın.

2. Sanal Ortam Oluşturun ve Aktive Edin:**
   Uygulamanızın(Örenk PyCharm) terminalini açın ve alttaki kodları sırasıyla çalıştırın
    
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

4.  **Bağımlılıkları Yükleyin:**
   gerekli paketleri doğrudan kurun
        ```bash
        pip install -U "protobuf>=5.29.1" streamlit langchain langchain-core langchain-community langchain-text-splitters langchain-google-genai google-generativeai faiss-cpu python-dotenv
        ```

5.  **Ortam Değişkenini Yapılandırın:**
terminalde aşağıdaki kodu çalıştırın. her yeni terminal oturumu için tekrarlamanız gerekir:
        * Windows (PowerShell)(PyCharm Terminali de aynı şekilde): `$env:GOOGLE_API_KEY = "AIzaSy..."`
        * Windows (CMD): `set GOOGLE_API_KEY=AIzaSy...`
        * Mac/Linux: `export GOOGLE_API_KEY=AIzaSy...`

6.  **Vektör Veritabanı Otomatik Oluşturulacak:** 
    * ⚠️ **Uyarı:** Bu ilk indeks oluşturma işlemi, anayasa metninin uzunluğuna bağlı olarak **birkaç dakika sürebilir** ve Google API kotanızı kullanır. Sonraki çalıştırmalar çok daha hızlı olacaktır.

7.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```

8.  Tarayıcınızı açın ve genellikle `http://localhost:8501` adresine gidin.

## Yapılandırma (Varsayılanlar - `app.py` içinde) ⚙️

* **Üretici Model (LLM):** `models/gemini-2.5-flash` (Sıcaklık `temperature=0`)
* **Embedding Modeli:** `models/text-embedding-004` (Google API)
* **Parçalama (Chunking):** 1000 karakter boyut, 200 karakter örtüşme (`overlap`)
* **Getirme (Retrieval) k:** En benzer 5 parça (`k=5`)

