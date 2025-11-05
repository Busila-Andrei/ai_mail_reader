# Email Q&A System

Sistem de interogare AI pentru emailuri folosind embeddings locale și FAISS.

## Caracteristici

- **Extragere text din emailuri**: Suportă fișiere `.eml`, `.msg`, text copiat, și screenshot-uri (OCR)
- **Indexare locală**: Folosește embeddings `all-MiniLM-L6-v2` și FAISS pentru stocare vectorială
- **Interogare AI**: Răspunde la întrebări despre emailuri folosind LLM local (Llama 3, Mistral)
- **Interfață consolă**: Interogări directe din terminal

## Flow

1. **Input**: Fișiere `.eml` / `.msg`, text copiat din mailuri, sau screenshot-uri (`.png`, `.jpg`)
2. **Extragere text**: Parsing emailuri sau OCR pentru imagini
3. **Preprocesare**: Împărțire în chunks (500-800 caractere) și normalizare text
4. **Indexare**: Generare embeddings și salvare în FAISS local
5. **Interogare**: Căutare semantică în FAISS + generare răspuns cu LLM
6. **Output**: Răspuns AI natural + surse (fișier, subiect, dată)

## Instalare

```bash
pip install -r requirements.txt
```

**Notă**: Pentru OCR, este necesară instalarea Tesseract OCR:
- Windows: Descarcă de la https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-ron`

## Utilizare

### 1. Indexare emailuri

```bash
# Indexare dintr-un director
python main.py index --input path/to/emails/

# Indexare un singur fișier
python main.py index --input email.eml

# Cu opțiuni personalizate
python main.py index --input emails/ --index-path data/my_index --chunk-size 800
```

### 2. Interogare

```bash
# Pornire mod chat
python main.py chat

# Cu index personalizat
python main.py chat --index-path data/my_index
```

În modul chat, introduce întrebări despre emailuri. Scrie `quit` pentru a ieși.

## Structură proiect

- `email_parser.py` - Parsing .eml/.msg și normalizare text
- `ocr_processor.py` - OCR pentru imagini
- `text_chunker.py` - Împărțire text în chunks
- `embedding_generator.py` - Generare embeddings cu sentence-transformers
- `faiss_indexer.py` - Stocare și căutare vectorială
- `llm_query.py` - Generare răspunsuri cu LLM local
- `console_chat.py` - Interfață consolă pentru interogare
- `main.py` - Punct de intrare principal

