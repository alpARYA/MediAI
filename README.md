# 🧠 MediAI: AI-Driven Symptom Checker and Health Advisor

MediAI is a real-time, AI-powered medical chatbot designed to assist users in symptom assessment and basic health triage. Built using cutting-edge NLP techniques and Retrieval-Augmented Generation (RAG), it combines the power of Mistral LLM, Langchain, Faiss vector store, and Streamlit for a responsive, privacy-conscious health advisory platform.

---

## 📌 Features

- ✅ Symptom-based Q&A using generative AI  
- 🔍 Context-aware responses using vector similarity search (FAISS)  
- 🧬 Mistral-7B-based medical chatbot via Hugging Face  
- 🔐 Local vector database with privacy focus (HIPAA/GDPR aligned)  
- 🖥️ User-friendly web interface built with Streamlit  
- 💬 Maintains conversation context with Langchain  
- 📂 Persistent memory with document retrieval support  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python, Langchain  
- **LLM:** Mistral-7B (via HuggingFace Hub)  
- **Vector Store:** FAISS  
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Environment Management:** dotenv  

---

## 📁 Project Structure

──.env # HuggingFace API Key
── medibot.py # Streamlit frontend application
── connect_memory_with_llm.py # Connects FAISS + Mistral for QA
── create_memory_for_llm.py # PDF ingestion, chunking, and vector store creation
── BT_40827_REPORT.pdf # Full project report
── BT40827_Research_Paper.pdf # Published research paper
── vectorstore/db_faiss/ # (Generated) FAISS index folder
── data/ # (Expected) folder for PDF files to embed

---

## 🚀 How to Run Locally

### 1. Clone the repository

git clone https://github.com/alpARYA/MediAI.git

cd MediAI

### 2.  Set up the environment
Install the required dependencies:

pip install -r requirements.txt

Create a .env file and add your HuggingFace API key:

HF_TOKEN="your_huggingface_api_key"

### 3. Ingest Documents (Optional)
Place your medical PDFs into the data/ folder and run:

python create_memory_for_llm.py

### 4. Launch the Chatbot

streamlit run medibot.py

---

#### 🔹 Example Query & Docs
```markdown
---

## 💡 Example Query

> **User:** I feel a sore throat and headache. What should I do?  
> **MediAI:** Based on the symptoms, you may be experiencing the early stages of a viral infection such as a cold or flu. It is advisable to rest, stay hydrated, and consult a healthcare provider if symptoms persist.

---
---

---
```

## ⚙️ Core Scripts

| File | Purpose |
|------|---------|
| `medibot.py` | Launches the Streamlit chatbot interface |
| `create_memory_for_llm.py` | Loads PDFs, splits text, and creates FAISS index |
| `connect_memory_with_llm.py` | CLI tool to test queries on vector-backed QA chain |
| `.env` | Stores HuggingFace API token securely |

---

## 🧪 Test Cases

| Input | Output |
|-------|--------|
| I have a sore throat | Suggest flu/cold and recommend doctor |
| What are COVID symptoms? | Fever, cough, fatigue, loss of taste/smell |
| How to handle high BP? | Lifestyle recommendations + doctor referral |
| My eyes hurt when I blink | Safety disclaimer + consult specialist |

---

## 📈 Performance Metrics

- ⏱️ **Response Time:** ~3 seconds/query  
- ✅ **Accuracy:** ~84% on 100 validated test cases  
- 🔐 **Privacy:** Fully local, HIPAA & GDPR aligned  

---

## 🧭 Future Enhancements

- 🔊 Voice-based interaction  
- 📱 Mobile app integration  
- 🌐 Multilingual support  
- 🩺 EHR (Electronic Health Record) integration  
- ⌚ Wearable health data support  

---

## 👨‍💻 Authors

- **Aryan Singh** – [thisisaryan1ia@gmail.com](mailto:thisisaryan1ia@gmail.com)  
- **Pavan Solanki** – [pawansk267@gmail.com](mailto:pawansk267@gmail.com)  

Under the guidance of **Dr. P. Sarvanan**, Galgotias University.

---

## 📝 License

This project is for academic and research use. For commercial licensing, contact the authors.

---

## 📎 Citation

> Aryan Singh, Pavan Solanki, *"MediAI: An AI-Powered Chatbot for Enhancing Healthcare Systems"*, ICICC 2025, Elsevier SSRN.


