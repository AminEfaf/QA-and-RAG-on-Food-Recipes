# Persian Question Answering System
A comprehensive Persian Natural Language Processing project implementing Question Answering systems through fine-tuning Large Language Models and Retrieval-Augmented Generation (RAG) approaches.

---

## Project Overview
This project presents a complete Persian Question Answering (QA) solution designed for:
- **Researchers** working on Persian NLP tasks
- **Students** learning about modern QA systems and RAG architectures
- **Developers** building Persian language applications
- **Educational purposes** demonstrating state-of-the-art NLP techniques

The project implements multiple approaches including LLM fine-tuning with LoRA/QLoRA, RAG-based retrieval systems with custom embedding models, and comprehensive evaluation metrics for Persian language understanding.

---

## Features

### Core Functionality
1. **LLM Fine-Tuning**
   - Fine-tuned Llama-3.2-1B-bnb-4bit model on PQuAD dataset
   - 4-bit quantization with LoRA/QLoRA for efficient training
   - Optimized for Persian question answering tasks
   - Evaluation metrics: F1-Score and Exact Match (EM)

2. **RAG Implementation**
   - Fine-tuned paraphrase-multilingual-MiniLM-L12-v2 embedding model
   - Dual chunking strategies: word-based and sentence-based
   - Multiple retrieval mechanisms: TF-IDF and BM25
   - Comprehensive performance analysis and comparison

3. **Semantic Similarity Analysis**
   - Cosine Similarity measurement between model outputs and ground truth
   - Mean Reciprocal Rank (MRR) calculation
   - Deep semantic understanding evaluation

4. **Advanced Retrieval System (Bonus)**
   - Fine-tuned distiluse-base-multilingual-cased-v2
   - Fine-tuned multilingual-e5-base and other embedding models
   - Vector database integration (FAISS, Chroma, LanceDB)
   - Advanced preprocessing pipelines for improved embedding space quality
   - Optimized retrieval speed and efficiency

5. **User Interface (Bonus)**
   - Interactive web interface for question answering
   - Real-time response generation
   - Built with Gradio/Streamlit/FastAPI
   - Clean and intuitive design for end users

### Evaluation Metrics
- **Accuracy Metrics**: F1-Score, Exact Match (EM)
- **Retrieval Metrics**: Precision, Recall, Hit@k
- **Semantic Similarity**: Cosine Similarity, Mean Reciprocal Rank (MRR)

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or Google Colab/Kaggle access
- 16GB+ RAM recommended for optimal performance

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/persian-qa-system.git
   cd persian-qa-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets**
   - PQuAD dataset from Hugging Face
   - Alternative: PersianQA dataset (for limited hardware scenarios)

4. **Run the experiments**
   - Part A: Fine-tune Llama model on PQuAD
   - Part B: Implement RAG with embedding fine-tuning
   - Part C: Calculate semantic similarity metrics
   - Bonus: Advanced embeddings and UI development

5. **Access the application**
   - Launch the web interface to interact with the QA system
   - Input questions and receive generated answers in real-time

---

## Project Structure

The project is organized into modular components for easy navigation and maintenance:

- **Data Processing**: Preprocessing pipelines for Persian text
- **Model Training**: Fine-tuning scripts for LLM and embeddings
- **Retrieval System**: Implementation of TF-IDF, BM25, and vector databases
- **Evaluation**: Comprehensive metric calculation and analysis
- **Interface**: User-facing application for question answering
- **Documentation**: Detailed reports on methodology and results
- 
---

## Technical Details

### Part A: LLM Fine-Tuning
- **Model**: Llama-3.2-1B with 4-bit quantization (unsloth/llama-3.2-1B-bnb-4bit)
- **Dataset**: PQuAD (alternative: PersianQA for hardware-limited scenarios)
- **Technique**: LoRA or QLoRA for parameter-efficient fine-tuning
- **Restriction**: Zero-Shot and Few-Shot methods are not permitted
- **Metrics**: F1-Score and Exact Match (EM)
- **Focus**: Resource-efficient training while maintaining high performance

### Part B: RAG System Implementation
**Embedding Model**
- Base model: paraphrase-multilingual-MiniLM-L12-v2
- Fine-tuned on project dataset
- Enhanced for Persian semantic understanding

**Chunking Strategies**
- Word-based chunking: Token-level document segmentation
- Sentence-based chunking: Sentence-level document segmentation
- Comparative analysis of both approaches

**Retrieval Methods**
- TF-IDF: Term frequency-inverse document frequency vectorization
- BM25: Probabilistic ranking function
- Performance comparison and analysis

**Evaluation Framework**
- F1-Score and Exact Match comparison
- Precision and Recall analysis
- Hit@k metrics for retrieval quality
- Comprehensive performance comparison between chunking strategies

### Part C: Semantic Similarity Analysis
- **Cosine Similarity**: Measures angular similarity between model output and ground truth
- **Mean Reciprocal Rank (MRR)**: Evaluates ranking quality of retrieved answers
- **Analysis**: Deep dive into semantic understanding capabilities

### Bonus Features
**Advanced Embedding Models**
- distiluse-base-multilingual-cased-v2: Distilled multilingual sentence encoder
- multilingual-e5-base: State-of-the-art multilingual embedding
- Custom embedding models: Additional experimentation with suitable models
- Comparative analysis: Performance evaluation across all models

**Preprocessing Enhancements**
- Text normalization for Persian language
- Stop word removal and stemming
- Character normalization and cleaning
- Embedding space quality optimization

**Vector Database Integration**
- FAISS: Facebook AI Similarity Search for efficient retrieval
- Chroma: Modern vector database for embeddings
- LanceDB: Columnar vector database
- Performance optimization: Speed and efficiency improvements

**Evaluation Metrics**
- Cosine Similarity across all models
- MRR comparison and analysis
- Precision, Recall, and Hit@k evaluation
- Comprehensive model comparison

**User Interface**
- Clean and intuitive design
- Question input interface
- Real-time answer generation
- Response visualization
- Built with modern web frameworks (Gradio/Streamlit/FastAPI)

---

## Datasets

### PQuAD (Persian Question Answering Dataset)
- **Source**: [Hugging Face - PQuAD](https://huggingface.co/datasets/Gholamreza/pquad)
- **Description**: Large-scale Persian question answering dataset
- **Format**: Context, Question, Answer triplets
- **Use Case**: Primary dataset for model training and evaluation

### PersianQA (Alternative Dataset)
- **Source**: [Hugging Face - PersianQA](https://huggingface.co/datasets/SajjadAyoubi/persian_qa)
- **Description**: Compact Persian QA dataset
- **Format**: Question-Answer pairs
- **Use Case**: Alternative for hardware-limited environments

---

## Models Used

### Language Models
- **Llama-3.2-1B-bnb-4bit**: [unsloth/llama-3.2-1B-bnb-4bit](https://huggingface.co/unsloth/llama-3.2-1B-bnb-4bit)
  - 4-bit quantized version for efficient training
  - Optimized for question answering tasks

### Retrieval Models
- **BM25**: [Qdrant/bm25](https://huggingface.co/Qdrant/bm25)
  - Traditional probabilistic ranking function
  - Baseline for retrieval comparison

### Embedding Models
- **MiniLM**: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  - Lightweight multilingual sentence encoder
  - Core embedding model for Part B

- **DistilUSE**: [sentence-transformers/distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
  - Distilled Universal Sentence Encoder
  - Bonus embedding model for comparison

- **E5-Base**: [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
  - State-of-the-art multilingual embeddings
  - Advanced bonus embedding model

---

## Authors

**Team Members**:
- Sobhan Heydarinezhad
- Mohammad Amin Efaf
- Arman Ghorbanpour 
