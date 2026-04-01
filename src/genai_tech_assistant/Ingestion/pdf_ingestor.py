from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
from pypdf import PdfReader
import re
from genai_tech_assistant.config import settings
from genai_tech_assistant.logging_config import get_logger

logger = get_logger(__name__)

# chunck object


@dataclass
class Documentchunk:
    id: str
    text: str
    metadata: Dict[str, Any]

# function to create chunks after cleaning the text


def split_into_chunks(
    text: str,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> list[str]:

    text = text.strip()
    if not text:
        return []

    # split the text based on sentences like a sentence ends with ? or . or !
    # in order to do that first it finds an empty space then checks the char before it
    # if the char is one of [.?!] it will split the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Clean up empty artifacts
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        # Start a new chunk with sentence i
        current_sentences = [sentences[i]]
        current_len = len(sentences[i])
        j = i + 1

        # Keep adding sentences while we stay under max_chars
        while j < n:
            next_sent = sentences[j]
            # +1 for space/newline between sentences
            candidate_len = current_len + 1 + len(next_sent)

            if candidate_len > max_chars:  # if this is true its gonna ignore appending the next sentence and goes to update chunk
                break

            current_sentences.append(next_sent)
            current_len = candidate_len
            j += 1

        # Join sentences into one chunk
        chunk_text = " ".join(current_sentences).strip()
        chunks.append(chunk_text)

        if j >= n:
            break

        # Overlap: move i so next chunk starts overlap_sentences before j
        # making sure the next chunk starts with the last sentence in previous chunk
        if overlap_sentences > 0:
            i = max(j - overlap_sentences, i + 1)
        else:
            i = j

    return chunks

# load pdf  clean and chunk it


def ingest_pdfs() -> List[Documentchunk]:

    pdf_dir: Path = settings.pdf_input_dir
    logger.info(f"Looking for PDFs in: {pdf_dir}")

    if not pdf_dir.exists():
        logger.warning(f"PDF directory does not exist: {pdf_dir}")
        return []

    pdf_paths = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_paths)} PDF files")

    chunks: List[Documentchunk] = []

    for pdf_path in pdf_paths:
        logger.info(f"Reading PDF: {pdf_path.name}")
        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path.name}: {e}")
            continue
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text()
            if not raw_text:
                logger.warning(
                    f"No text extracted from {pdf_path.name} page {page_number}")
                continue
            cleaned_txt = " ".join(raw_text.split())
            logger.info(
                f"Extracted ~{len(cleaned_txt)} characters from {pdf_path.name} page {page_number}"
            )
            page_chunks = split_into_chunks(
                cleaned_txt, max_chars=1000, overlap_sentences=1)
            logger.info(
                f"Created {len(page_chunks)} chunks from {pdf_path.name} page {page_number}"
            )
            # combinging each chunk with its id and metadata
            for chunk_index, chunk_text in enumerate(page_chunks):
                chunk_id = f"{pdf_path.name}_page_{page_number}_chunk_{chunk_index}"
                metadata = {
                    "source_file": pdf_path.name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text),
                }
                chunk_obj = Documentchunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                )

                chunks.append(chunk_obj)
    logger.info(f"No of chunks created: {len(chunks)}")
    return chunks
