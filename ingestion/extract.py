import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    
    """
    Extract raw text from a PDF file using PyMuPDF.
    Returns the full text as a single string.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n\n--- Page {page_num + 1} ---\n\n"
        full_text += text

    doc.close()
    return full_text


def extract_all_pdfs(raw_dir: str, processed_dir: str):
    """
    Loop through all PDFs in raw_dir, extract text,
    and save each as a .txt file in processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(raw_dir, filename)
            print(f"Extracting: {filename}")

            text = extract_text_from_pdf(pdf_path)

            output_filename = filename.replace(".pdf", ".txt")
            output_path = os.path.join(processed_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  ✓ Saved to {output_path}")


if __name__ == "__main__":
    extract_all_pdfs("data/raw", "data/processed")