# app.py (Güncellenmiş - API Embedding Kullanımı)
import streamlit as st
import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings # Yerel model için - ARTIK KULLANILMIYOR
from langchain_google_genai import GoogleGenerativeAIEmbeddings # <-- API için EKLENDİ
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
# from huggingface_hub import snapshot_download # <-- Artık GEREKMİYOR

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Anayasa RAG Botu", page_icon="🇹🇷", layout="wide")

# --- 1. Konfigürasyon ve API Anahtarı ---
logger.info("Uygulama başlatılıyor...")
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("API Anahtarı ortam değişkeninden başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"API anahtarı yapılandırılırken hata oluştu: {e}", exc_info=True)
        st.error(f"Google API anahtarı yapılandırılırken bir hata oluştu: {e}")
        st.stop()
else:
    logger.error("HATA: Google API anahtarı 'GOOGLE_API_KEY' ortam değişkeninde bulunamadı!")
    st.error("HATA: Google API anahtarı 'GOOGLE_API_KEY' ortam değişkeninde bulunamadı!")
    st.info("Lütfen çalıştırmadan önce terminalde API anahtarınızı ayarlayın...") # Mesaj aynı
    st.stop()

# Lokal Dosya Yolları
# MODEL_DIR = "bge-m3" # <-- Artık GEREKMİYOR
# FAISS indeks klasör adı için yeni bir isim verelim ki eskiyle karışmasın
FAISS_INDEX_DIR = "faiss_index_anayasa_google_emb" # <-- İsim değişti
DATA_FILE = "anayasa.txt"
# Google'ın önerilen embedding model adı
GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004" # Veya "models/embedding-001"

# --- 2. Gerekli Fonksiyonları ve Nesneleri Yükleme (Cache ile) ---

@st.cache_resource # Embedding API istemcisini sadece bir kez kur
def load_embedding_model(api_key, model_name):
    logger.info(f"Google Embedding API istemcisi ('{model_name}') yükleniyor...")
    try:
        # API anahtarının zaten configure edildiğini varsayıyoruz
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        # Hızlı bir test yapalım (opsiyonel ama iyi fikir)
        _ = embeddings.embed_query("test")
        logger.info("Google Embedding API istemcisi başarıyla yüklendi ve test edildi.")
        return embeddings
    except Exception as e:
        st.error(f"Google Embedding API istemcisi yüklenirken/test edilirken hata: {e}")
        logger.error(f"Google Embedding API istemcisi yüklenirken/test edilirken hata: {e}", exc_info=True)
        return None

@st.cache_resource # FAISS indeksini sadece bir kez yükle (veya oluştur)
def load_or_create_vector_store(index_dir, data_file, _embeddings):
    logger.info(f"FAISS index '{index_dir}' kontrol ediliyor...")
    if _embeddings is None:
        st.error("Embedding modeli yüklenemediği için FAISS işlemi yapılamıyor.")
        logger.error("Embedding modeli yüklenemediği için FAISS işlemi yapılamıyor.")
        return None

    if os.path.isdir(index_dir):
        logger.info("Önceden oluşturulmuş FAISS indeksi diskten yükleniyor...")
        try:
            # API embedding modeli ile oluşturulmuş indeksi yükle
            vector_store = FAISS.load_local(index_dir, _embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS indeksi başarıyla yüklendi.")
            st.success("Vektör veritabanı hazırlandı.")
            return vector_store
        except Exception as e:
            st.warning(f"FAISS indeksi '{index_dir}' yüklenirken hata oluştu: {e}. İndeks yeniden oluşturulacak.")
            logger.warning(f"FAISS indeksi '{index_dir}' yüklenirken hata oluştu: {e}. İndeks yeniden oluşturulacak.")
    else:
         logger.warning(f"FAISS indeksi klasörü '{index_dir}' bulunamadı. Veri işlenip SIFIRDAN oluşturulacak...")
         st.warning(f"FAISS indeksi klasörü '{index_dir}' bulunamadı veya yüklenemedi. Veri işlenip SIFIRDAN oluşturuluyor...")
         st.warning("Bu işlem BİR KEZ API çağrıları nedeniyle zaman alabilir ve API kotanızı kullanabilir...")

    # --- İndeks yoksa veya yüklenemediyse SIFIRDAN OLUŞTUR ---
    progress_bar = st.progress(0, text="Vektör veritabanı oluşturuluyor (API çağrıları yapılıyor)...")
    try:
        if not os.path.exists(data_file):
            st.error(f"Veri dosyası '{data_file}' bulunamadı!")
            logger.error(f"Veri dosyası '{data_file}' bulunamadı!")
            progress_bar.progress(100, text="Hata: Veri dosyası bulunamadı!")
            return None

        logger.info(f"Veri yükleniyor: {data_file}")
        loader = TextLoader(data_file, encoding="utf-8")
        documents = loader.load()
        progress_bar.progress(10, text="Veri yüklendi.")

        logger.info("Metin parçalanıyor...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"{len(docs)} parça oluşturuldu.")
        progress_bar.progress(30, text="Metin parçalandı.")

        logger.info(f"Vektör veritabanı (FAISS) oluşturuluyor - HER PARÇA İÇİN API ÇAĞRISI YAPILACAK ({len(docs)} adet)...")
        # --- BURASI DEĞİŞTİ: FAISS.from_documents artık API çağrıları yapacak ---
        vector_store = FAISS.from_documents(docs, _embeddings)
        # -------------------------------------------------------------------
        logger.info("Vektör veritabanı oluşturuldu (API çağrıları tamamlandı).")
        progress_bar.progress(80, text="Vektörler API'den alındı.")

        logger.info(f"FAISS indeksi '{index_dir}' yoluna kaydediliyor...")
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        logger.info("FAISS indeksi kaydedildi.")
        progress_bar.progress(100, text="Vektör veritabanı kaydedildi.")
        st.success("Vektör veritabanı başarıyla oluşturuldu ve kaydedildi.")
        return vector_store

    except Exception as e:
        st.error(f"FAISS oluşturulurken/kaydedilirken HATA: {e}")
        logger.error(f"FAISS oluşturulurken/kaydedilirken HATA: {e}", exc_info=True)
        progress_bar.progress(100, text="Hata: Vektör veritabanı oluşturulamadı!")
        return None

@st.cache_resource # Gemini LLM'i sadece bir kez kur
def load_llm():
    # ... (Bu fonksiyon aynı) ...
    logger.info("Gemini LLM ('models/gemini-2.5-flash') yükleniyor...")
    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
        logger.info("Gemini LLM başarıyla yüklendi.")
        return llm
    except Exception as e:
        st.error(f"Gemini LLM yüklenirken HATA: {e}")
        logger.error(f"Gemini LLM yüklenirken HATA: {e}", exc_info=True)
        return None

# --- 3. Nesneleri Yükle ---
# Google Embedding API istemcisini yükle
embeddings = load_embedding_model(GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL)
# FAISS indeksini yükle veya API kullanarak oluştur
vector_store = load_or_create_vector_store(FAISS_INDEX_DIR, DATA_FILE, embeddings)
llm = load_llm()

# --- 4. RAG Zincirini Kur ---
# ... (Bu kısım aynı) ...
rag_chain = None
if embeddings and vector_store and llm:
    # ... (prompt ve chain kurulumu aynı) ...
    try:
        prompt_template_str = "Aşağıdaki ilgili metin parçalarını kullanarak soruyu cevapla. Sadece verilen metinlere dayanarak cevap ver, dışarıdan bilgi ekleme.\n\nİlgili Metinler:\n{context}\n\nSoru: {question}\nCevap:"
        prompt = ChatPromptTemplate.from_template(prompt_template_str)

        rag_chain = RunnableSequence(
            prompt,
            llm,
            StrOutputParser()
        )
        logger.info("RAG zinciri başarıyla kuruldu.")
    except Exception as e:
        st.error(f"RAG zinciri kurulurken hata: {e}")
        logger.error(f"RAG zinciri kurulurken hata: {e}", exc_info=True)

else:
    st.error("Gerekli bileşenler (Embedding, Vector Store veya LLM) yüklenemediği için RAG zinciri kurulamadı. Terminal loglarını kontrol edin.")
    logger.error("Gerekli bileşenler yüklenemediği için RAG zinciri kurulamadı.")


# --- 5. Streamlit Arayüzü ---
# ... (Bu kısım aynı) ...
st.title("🇹🇷 Anayasa RAG Sohbet Botu (API Embedding)")
st.caption("1982 Türkiye Anayasası (Güncel Hali) ile ilgili sorularınızı sorun.")

# ... (chat history ve input/output kısımları aynı) ...
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Anayasa hakkında ne merak ediyorsun?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if rag_chain:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Cevap aranıyor... 🧠"):
                try:
                    logger.info(f"Kullanıcı sorgusu: {prompt}")
                    # 1. Benzer belgeleri bul (Artık API embedding'leri ile)
                    relevant_docs = vector_store.similarity_search(prompt, k=5)
                    logger.info(f"{len(relevant_docs)} adet ilgili belge bulundu.")
                    # ... (context oluşturma, chain.invoke, response gösterme aynı) ...
                    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                    logger.info("LLM'e istek gönderiliyor...")
                    inputs = {"context": context_text, "question": prompt}
                    response = rag_chain.invoke(inputs)
                    logger.info("LLM'den cevap alındı.")
                    full_response = response

                except Exception as e:
                    full_response = f"Üzgünüm, bir hata oluştu: {e}"
                    logger.error(f"Cevap üretilirken bir hata oluştu: {e}", exc_info=True)

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("Uygulama başlatılamadığı için cevap verilemiyor. Lütfen terminal loglarını kontrol edin.")