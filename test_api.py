# Run with: python test_api.py  (needs httpx, which FastAPI's TestClient uses)
import io
import os

os.environ["GROQ_API_KEY"] = "test-key-not-real"  # no Groq calls happen here

import PyPDF2
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def upload(name, data):
    return client.post("/api/upload", files={"file": (name, data)})


def blank_pdf():
    # A page with no text is where extract_text() can come back empty
    writer = PyPDF2.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# Upload size limit, including the exact boundary
assert upload("big.txt", b"x" * (api.MAX_UPLOAD_BYTES + 1)).status_code == 413
assert upload("edge.txt", b"x" * api.MAX_UPLOAD_BYTES).status_code == 200

# Extracted text is truncated
assert upload("a.txt", b"x" * 20000).json() == {"extracted_text": "x" * api.MAX_TEXT_CHARS}

# PDFs still parse, and a corrupt one returns an error instead of crashing
r = upload("blank.pdf", blank_pdf())
assert r.status_code == 200 and "error" not in r.json(), r.text
r = upload("bad.pdf", b"not a pdf")
assert r.status_code == 200 and r.json()["extracted_text"] == "" and "error" in r.json()

# CORS answers a browser preflight from another domain
r = client.options(
    "/api/chat",
    headers={"Origin": "https://example.vercel.app", "Access-Control-Request-Method": "POST"},
)
assert r.status_code == 200 and "access-control-allow-origin" in r.headers

print("ok")
