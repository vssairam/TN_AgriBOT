# TN AgriBOT Assistant — PDF RAG with Streamlit, LangChain, FAISS, and OpenAI

TN AgriBOT is a Streamlit Farmer Assistant that answers questions across **multiple uploaded PDF documents**. It extracts PDF text, splits it into overlapping chunks, embeds the chunks with OpenAI embeddings, and stores all chunks in a single FAISS vector database. Answers include source PDF names and page numbers, with expandable excerpts to help users validate the response.

## Features

- Upload several PDFs in one session.
- Combine all uploaded documents into one FAISS index.
- Chunking with `RecursiveCharacterTextSplitter` (1,000 characters, 180-character overlap).
- Show the total number of chunks indexed.
- Retrieve relevant passages and answer using an OpenAI chat model through LangChain.
- Display source filename and page number, plus the retrieved text for verification.
- Chat history maintained in Streamlit session state.
- Clear chat/search results without rebuilding the index.
- Reset the knowledge base when you want to start over.
- Agricultural green/cream theme with farm imagery.
- API key read from the `OPENAI_API_KEY` environment variable; no key is hard-coded.

## Files

```text
TN_AgriBOT_Project/
├── TN_AgriBOT_Assistant.py
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- An OpenAI API key with API access and billing configured
- Internet access while generating embeddings and answers
- Text-readable PDFs (scanned/image-only PDFs require OCR, which this version does not include)

## 1. Create a virtual environment (Windows PowerShell)

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

## 2. Install packages

Create a `requirements.txt` file with:

```text
streamlit>=1.36.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-text-splitters>=0.3.0
faiss-cpu>=1.8.0
pypdf>=4.0.0
```

Then install:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Set the OpenAI API key as an environment variable

The app reads the key from the process environment. Do not paste your key into the Python source code or commit it to version control.

**For the current PowerShell window only:**

```powershell
$env:OPENAI_API_KEY = "your_actual_api_key"
```

**Persist it for your Windows user account:**

```powershell
setx OPENAI_API_KEY "your_actual_api_key"
```

After using `setx`, open a **new** PowerShell window so the new environment variable is available. Avoid sharing screenshots or logs that expose your key.

To verify that the variable is set without printing the secret:

```powershell
if ($env:OPENAI_API_KEY) { "OPENAI_API_KEY is set" } else { "OPENAI_API_KEY is missing" }
```

## 4. Run the application

From the project folder, with the virtual environment active and the API key set:

```powershell
streamlit run TN_AgriBOT_Assistant.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`. Open it in your browser.

## 5. How to use TN AgriBOT

1. In the sidebar, select multiple PDF documents.
2. Click **Build / Replace Knowledge Base**.
3. Wait for PDF text extraction, chunking, embedding, and FAISS indexing to finish.
4. Confirm the **Text chunks indexed** count in the sidebar.
5. Ask a question in the chat input.
6. Review the answer and its citations, such as `soil_guide.pdf, page 4`.
7. Expand each source to inspect the retrieved passage from the original PDF.
8. Click **Clear chat / search results** to start a new conversation while retaining the current index.
9. Click **Reset knowledge base** to clear the index and chat.

To replace the indexed documents, select the new set of PDFs and click **Build / Replace Knowledge Base** again. The newly built index replaces the previous one.

## Example questions

- What soil preparation steps are described in the uploaded guides?
- Summarize the irrigation recommendations.
- Which symptoms are mentioned for the crop disease?
- Compare the fertilizer guidance in the documents.
- What precautions does the manual mention before applying a farm input?
- Which document and page discuss seed treatment?

The quality of answers depends on the quality and coverage of the uploaded documents. If the documents do not contain an answer, the assistant is instructed to say so instead of guessing.

## Architecture

```text
Multiple PDF uploads
        |
        v
PyPDFLoader (page-level text + metadata)
        |
        v
RecursiveCharacterTextSplitter
        |
        v
OpenAI text-embedding-3-small
        |
        v
One FAISS vector store in Streamlit session state
        |
User question
        |
        v
FAISS similarity search (top 5 chunks)
        |
        v
OpenAI gpt-4o-mini via LangChain
        |
        v
Answer + source filenames + page numbers + evidence excerpts
```

## Important implementation notes

- **Page numbers:** PDF loader metadata uses zero-based page indices. The application converts these to human-friendly page numbers starting at 1.
- **Chunk count:** The displayed number is the count of non-empty text chunks passed to FAISS.
- **Session state:** The FAISS index and chat history are held in Streamlit session state. They are not persisted across server restarts or browser sessions.
- **Scanned PDFs:** This version uses text extraction and does not include OCR. Scanned PDFs may produce no readable text and will need OCR preprocessing.
- **API costs:** Embedding documents and generating answers use the OpenAI API and may incur charges.
- **Agricultural safety:** TN AgriBOT answers from the uploaded material and can still make mistakes. Verify important crop, pesticide, fertilizer, livestock, and food-safety decisions with official local agricultural guidance and product labels.

## Troubleshooting

### `OPENAI_API_KEY was not found`
Set the environment variable in the same terminal from which you start Streamlit. If you used `setx`, open a new terminal.

### `401 invalid_api_key`
Check that the key is correct, active, and available to the API project/account. Do not include extra spaces or quote characters inside the value.

### API quota or billing errors
Check the API account's usage limits and billing configuration. A ChatGPT subscription and OpenAI API billing are separate.

### No readable text found
The PDFs may be scanned images. Run OCR on them first, then upload the searchable PDFs.

### Import errors after installation
Activate the intended virtual environment and run:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

## References

- [LangChain documentation](https://docs.langchain.com/)
- [LangChain OpenAI integration](https://docs.langchain.com/oss/python/integrations/providers/openai)
- [LangChain FAISS integration](https://docs.langchain.com/oss/python/integrations/vectorstores/faiss)
- [Streamlit documentation](https://docs.streamlit.io/)
- [OpenAI API documentation](https://platform.openai.com/docs/)

## License

For learning and demonstration use. Review dependencies and deployment/security requirements before using this application in a production setting.
