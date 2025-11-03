# Dependencies

This project uses the following runtime dependencies (by import):

- Core Python: `os`, `re`, `json`, `argparse`, `datetime`, `pathlib`, `typing`, `xml.etree.ElementTree`
- HTTP/Retry: `requests`, `urllib3`
- Data: `pandas`
- HTML/XML Parsing: `beautifulsoup4` (module `bs4`), `lxml`
- Windows/Outlook Automation: `pywin32` (modules `win32com.client`, `pywintypes`)
- Env management: `python-dotenv` (module `dotenv`)
- SSH: `paramiko`
- AI (optional): `openai` (only when configured), local Ollama HTTP API
- Local memory helper: `rag_memory` (local module expected in project path)

Notes:
- If you use the optional OpenAI provider, ensure `openai` is installed and `OPENAI_API_KEY` is set.
- On Windows, `pywin32` is required for Outlook access.
- `lxml` is recommended for fast HTML parsing in BeautifulSoup.

## Recommended requirements.txt generation

To capture your exact environment versions, generate `requirements.txt` from the active virtual environment:

```bash
pip freeze > requirements.txt
```

Run it from the project root with the intended virtual environment activated.

