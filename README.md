# BLOCHAIN_RAG_FOR_LASTDAY_EXAM_PREP

## Introduction

Welcome to **BLOCHAIN_RAG_FOR_LASTDAY_EXAM_PREP**, an innovative project that combines **AI** and **Web3** technologies to create a unique platform for intelligent document querying and blockchain-based verification. This project is the first of its kind, integrating **Retrieval-Augmented Generation (RAG)** with **Smart Contracts** to address real-world challenges in decentralized systems.

Our primary goal is to demonstrate how AI can enhance the usability and functionality of Web3 platforms, paving the way for future advancements in decentralized ecosystems.

---

## Phase 1: Architecture Overview

The following diagram outlines the architecture of Phase 1, showcasing the technical complexity, usability, and real-world impact of our system:

![Architecture Diagram](https://drive.google.com/file/d/1WLrnT10g_drPOxdR7Ai2mcykGmfU3fsf/view?usp=sharing)

### Key Components:

#### **User Interface**
- **Innovation & Usability**: A drag-and-drop interface allows users to upload PDFs or DOCX files (up to 25MB) seamlessly.
- **Formats Supported**: PDF, DOCX.
- **Real-World Impact**: Simplifies document processing for end-users.

#### **PDF Processing**
- **Technical Complexity**: Utilizes advanced OCR tools like PyPDF2 and Tesseract to extract text and parse metadata.
- **Innovation**: Enables accurate data extraction from complex documents.

#### **Text Chunking**
- **Technical Complexity**: Implements recursive methods using Langchain Splitters and NLTK for efficient text segmentation.
- **Chunk Size**: 1024 tokens.
- **Overlap**: 200 tokens.
- **Real-World Impact**: Prepares data for embedding generation with high accuracy.

#### **Embedding Generation**
- **Technical Complexity**: Generates embeddings using cutting-edge models like OpenAI Ada and Sentence Transformers.
- **Model Used**: `text-embedding-ada-002`.
- **Dimensions**: 1536.
- **Index Type**: IVF (Inverted File Index).
- **Metric Used**: L2 (Euclidean Distance).
- **Innovation & Creativity**: Provides robust vector representations for similarity search.

#### **FAISS Vector Store**
- **Technical Complexity**: Stores embeddings for efficient similarity search.
- **Index Parameters**:
  - `nlist`: 100
  - Similarity Threshold: 0.75
- **Real-World Impact**: Enables fast and accurate retrieval of relevant document sections.

#### **SHA-256 Hashing**
- **Technical Complexity & Innovation**: Ensures content integrity through hashing of original PDFs and text content.
- **Usability & Design**: Metadata storage enhances verification processes.

#### **RetrievalQA LLM**
- **Innovation & Creativity**: Uses advanced language models like GPT-4, Claude, or Llama for intelligent querying.
- **Context Window Size**: 32K tokens.
- **Top-K Retrieval**: Up to 5 documents.
- **Real-World Impact**: Provides accurate responses with linked citations, confidence scores, and page references.

#### **Blockchain Integration**
1. Blockchain Manager:
   - Integrates Ethereum/Polygon networks for decentralized storage and validation.
   - Supports ERC-1155 standards optimized for record storage.
   - Gas-efficient transactions.
2. Smart Contract:
   - Solidity-based contracts ensure secure record storage and verification.
   - Real-World Impact: Guarantees trust in decentralized systems.

#### **Answer Generation**
- Innovation & Usability:
  - Direct responses formatted with citations, confidence scores, and page references.
  - Enhances user experience by providing clear and reliable answers.

---

## Innovation & Creativity

This project stands out due to its unique integration of AI-powered document querying with blockchain verification mechanisms. By combining RAG with Smart Contracts, we have created a system that addresses both usability and security concerns in decentralized environments. The architecture leverages cutting-edge AI models and blockchain technologies to deliver innovative solutions that are practical and impactful.

---

## Technical Complexity

The system incorporates advanced components such as:
1. Recursive text chunking for efficient data processing.
2. Embedding generation using state-of-the-art models optimized for similarity search.
3. GPU-based parallel processing for handling large-scale operations.
4. Blockchain integration with gas-efficient smart contracts ensures secure data storage and retrieval.

These features highlight the technical depth of the project while ensuring scalability and reliability.

---

## Usability & Design

The user-friendly interface simplifies document uploading and querying, making it accessible even to non-technical users. The system provides clear answers with linked sources, confidence scores, and page references, ensuring transparency and ease of use.

---

## Real-World Impact & Problem Statement Clarity

This project addresses critical challenges in decentralized ecosystems:
1. Efficient document querying in blockchain environments.
2. Secure content verification using hashing mechanisms.
3. Bridging the gap between AI capabilities and Web3 functionalities.

By solving these problems, BLOCHAIN_RAG_FOR_LASTDAY_EXAM_PREP demonstrates its potential to transform industries reliant on decentralized systems.

---

## Current Status

We have successfully developed **90% of Phase 1**, including all core functionalities outlined above. However, deployment remains a challenge due to the lack of platforms supporting parallel processing with GPU resources required for real-time operations.

---

## How to Run the Application

To experience the system, follow these steps:

1. Visit the Hugging Face Spaces link: [BLOCHAIN_RAG_FOR_LASTDAY_EXAM_PREP](https://huggingface.co/spaces/vikramronavrsc/BLOCHAIN_RAG_FOR_LASTDAY_EXAM_PREP).
2. Click on **Initialize System** to set up the environment.
3. Select **Simulate Connection** to connect to Ethereum or any other decentralized blockchain platform's mainnet.
4. Upload your files and start querying!

---

## Future Development

### Phase 2 Goals:
1. Develop deployment solutions that support GPU-based parallel processing.
2. Expand compatibility with more decentralized platforms beyond Ethereum/Polygon networks.
3. Incorporate additional AI models for enhanced query accuracy.

---

## Contribution & Acknowledgements

We welcome contributions from developers passionate about AI-Web3 integration! Feel free to explore our repository, suggest improvements, or collaborate on future phases.

Special thanks to our team for their dedication in bringing this innovative concept to life!
