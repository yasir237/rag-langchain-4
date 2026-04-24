# LangChain ile Metin Bölme ve Embedding
> Üretilen açıklamayı anlamlı chunk'lara bölme ve Gemini ile vektöre çevirme — RAG serisinin 4. adımı

[![Colab'da Aç](https://img.shields.io/badge/Colab'da%20Aç-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yasir237/rag-langchain-4/blob/main/rag_langchain_4.ipynb)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)

---

## Problem

LLM'in ürettiği uzun metni olduğu gibi vektör store'a atarsan anlam kaybı olur. Model tüm metne tek bir vektör üretir — bu vektör hiçbir şeyi tam temsil etmez. Üstelik farklı konu bloklarını birbirinden ayırt edemez.

## Çözüm

Metni önce anlamlı parçalara (chunk) bölmek, ardından her parçayı ayrı ayrı embed etmek gerekir. Böylece vektör store'da her chunk kendi anlamını taşır ve arama sırasında doğru parça bulunur. Bu adım, RAG pipeline'ının **retrieval** kısmının kalitesini doğrudan belirler.

---

## Pipeline Mimarisi

```
LLM Çıktısı (response3 — uzun Türkçe metin)
        │
        ▼
┌───────────────────────────────────────┐
│           Ön İşleme                   │
│  Markdown başlıkları temizlenir       │
│  Duplicate chunk'lar kaldırılır       │
│  Çok kısa / anlamsız parçalar atılır  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│     RecursiveCharacterTextSplitter    │
│  chunk_size=300                       │
│  chunk_overlap=50                     │
│  separators=["\n\n", "\n", ". ", " "] │
└───────────────────────────────────────┘
        │  ~6-8 anlamlı chunk
        ▼
┌───────────────────────────────────────┐
│      GoogleGenerativeAIEmbeddings     │
│  model: gemini-embedding-001          │
│  boyut: 3072                          │
└───────────────────────────────────────┘
        │
        ▼
  Her chunk → 3072 boyutlu vektör
```

| Bileşen | Görevi |
|---|---|
| `RecursiveCharacterTextSplitter` | Metni önce paragrafa, sonra cümleye, sonra boşluğa göre böler |
| `chunk_size` | Her parçanın maksimum karakter sayısı |
| `chunk_overlap` | Parçalar arasında tekrar eden karakter sayısı — bağlam kopmasın diye |
| `separators` | Bölme öncelik sırası — cümle ortasından kesmeyi engeller |
| `embed_query` | Tek bir metni vektöre çevirir |
| `embed_documents` | Birden fazla chunk'ı toplu embed eder |

---

## Kullanılan Teknolojiler

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_Embedding-4285F4?style=flat&logo=google&logoColor=white)
![Llama](https://img.shields.io/badge/Llama_3.1_8B-0467DF?style=flat&logo=meta&logoColor=white)
![Python](https://img.shields.io/badge/Python_3-3776AB?style=flat&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

---

## Kurulum

```bash
pip install langchain langchain-core langchain-groq
pip install langchain-text-splitters langchain-google-genai
```

### API Anahtarları

Google Colab **Secrets** sekmesine ekle:
- `GROQ_API_KEY` → [console.groq.com](https://console.groq.com)
- `GOOGLE_API_KEY` → [aistudio.google.com](https://aistudio.google.com)

---

## Kullanım

```python
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Metni temizle
def metni_temizle(metin):
    metin = re.sub(r'\*\*.*?\*\*\n?', '', metin)  # Markdown başlıklarını kaldır
    metin = re.sub(r'\n{3,}', '\n\n', metin)
    return metin.strip()

temiz_metin = metni_temizle(explanation)

# 2. Chunk'lara böl
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)
texts = text_splitter.create_documents([temiz_metin])

# 3. Anlamsız chunk'ları temizle
texts = [
    t for t in texts
    if len(t.page_content.strip()) > 30
    and not t.page_content.strip().startswith(".")
    and not t.page_content.strip().endswith(":")
]

# 4. Duplicate chunk'ları kaldır
benzersiz, gorülen = [], set()
for chunk in texts:
    anahtar = chunk.page_content[:50].strip()
    if anahtar not in gorülen:
        gorülen.add(anahtar)
        benzersiz.append(chunk)
texts = benzersiz

# 5. Embed et
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

query_result = embeddings.embed_query(texts[0].page_content)
print(f"Vektör boyutu: {len(query_result)}")  # 3072

all_embeddings = embeddings.embed_documents([t.page_content for t in texts])
```

---

## Neden Gemini Embedding?

| | HuggingFace MiniLM | OpenAI ada-002 | Gemini Embedding |
|---|---|---|---|
| Vektör boyutu | 384 | 1536 | 3072 |
| Türkçe desteği | ✅ Çok iyi | ⚠️ Orta | ✅ İyi |
| Ücret | Ücretsiz | Ücretli | Ücretsiz limit |
| Kurulum | Local model indirir | API key yeter | API key yeter |

Öğrenme aşaması için Gemini; ücretsiz, kurulumu kolay ve 3072 boyutlu vektörüyle yüksek kaliteli bir seçim.

---

## Seri İçindeki Yeri

Bu notebook, LangChain ile kurulan RAG serisinin **4. adımıdır.**

```
[1] ✅ Mesaj yapısı ve LLM bağlantısı
[2] ✅ PromptTemplate ile şablonlu prompt
[3] ✅ Çoklu zincir kurma ve zincirleri bağlama
[4] ✅ Metin bölme ve embedding                  ← bu repo
[5]    Vektör store ve benzerlik araması
```

Her adım bir sonrakine köprü kuruyor.  
Serinin tamamını takip etmek için LinkedIn profilimi ziyaret edebilirsin 👇

---

## Bağlantı

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yasir237)

---

## Lisans

MIT
