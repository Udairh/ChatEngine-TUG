import pdfplumber

def parse_pdfs(pdf_files):
    documents = {}
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
            documents[pdf_file] = text
    return documents

if __name__ == "__main__":
    pdf_files = ["data/goog.pdf", "data/tesla.pdf", "data/uber.pdf"]
    documents = parse_pdfs(pdf_files)
    for name, content in documents.items():
        print(f"Parsed content from {name}: {content[:500]}...")