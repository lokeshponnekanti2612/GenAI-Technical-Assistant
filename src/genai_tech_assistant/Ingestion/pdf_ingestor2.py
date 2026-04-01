from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter, defaultdict
import re

from pypdf import PdfReader

from genai_tech_assistant.config import settings
from genai_tech_assistant.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Documentchunk:
    id: str
    text: str
    metadata: Dict[str, Any]


def normalize_line(line: str) -> str:

    # Normalize a single line for comparison and cleaning.

    if not line:
        return ""

    text = line.strip()

    # Repair common line-break hyphenation:
    # "trouble- shooting" to "troubleshooting"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def line_content_stats(line: str) -> Dict[str, float]:

    # Compute lightweight shape/content features for a line.

    if not line:
        return {
            "length": 0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "upper_ratio": 0.0,
            "punct_ratio": 0.0,
            "word_count": 0,
        }

    length = len(line)
    alpha_count = sum(ch.isalpha() for ch in line)
    digit_count = sum(ch.isdigit() for ch in line)
    upper_count = sum(ch.isupper() for ch in line if ch.isalpha())
    punct_count = sum(not ch.isalnum() and not ch.isspace() for ch in line)
    word_count = len(line.split())

    alpha_ratio = alpha_count / length if length else 0.0
    digit_ratio = digit_count / length if length else 0.0
    upper_ratio = upper_count / alpha_count if alpha_count else 0.0
    punct_ratio = punct_count / length if length else 0.0

    return {
        "length": length,
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "upper_ratio": upper_ratio,
        "punct_ratio": punct_ratio,
        "word_count": word_count,
    }


def is_intrinsically_bad_line(line: str) -> bool:

    # Remove obvious junk lines, but avoid document-specific hardcoded text.

    if not line:
        return True

    stats = line_content_stats(line)

    # Extremely short lines are rarely useful
    if stats["length"] <= 2:
        return True

    # URL-only / mostly URL-ish lines are junk
    low = line.lower()
    if low.startswith("http://") or low.startswith("https://") or low.startswith("www."):
        return True

    # Very punctuation-heavy tiny fragments are junk
    if stats["length"] < 12 and stats["punct_ratio"] > 0.25:
        return True

    return False


def is_probable_repeated_boilerplate(
    line: str,
    page_frequency: int,
    total_pages: int,
    top_occurrences: int,
    bottom_occurrences: int,
) -> bool:

    # Decide whether a line behaves like repeated header/footer/navigation boilerplate.

    if not line:
        return True

    if total_pages < 4:
        return False

    stats = line_content_stats(line)

    # Must repeat across enough pages to be suspicious
    repeated_enough = page_frequency >= max(3, total_pages // 4)
    if not repeated_enough:
        return False

    #  repeated lines near top/bottom are suspicious
    positional_bias = (top_occurrences +
                       bottom_occurrences) >= max(2, page_frequency // 2)

    # Short/medium repeated lines are strong boilerplate candidates
    shortish = stats["length"] <= 90

    # header-like shape:
    # high uppercase or low natural-sentence feel
    navigation_like = (
        stats["upper_ratio"] > 0.55
        or (stats["word_count"] <= 8 and stats["punct_ratio"] < 0.10 and stats["alpha_ratio"] > 0.55)
    )

    # Strong rule: repeated , positional , short
    if positional_bias and shortish:
        return True

    # Backup rule: repeated , short , navigation-like
    if shortish and navigation_like:
        return True

    return False


def clean_page_lines(
    page_lines: List[str],
    repeated_line_counts: Counter,
    line_top_counts: Counter,
    line_bottom_counts: Counter,
    total_pages: int,
) -> str:

    # Clean one page's lines using:

    cleaned_lines: List[str] = []

    for raw_line in page_lines:
        line = normalize_line(raw_line)

        if not line:
            continue

        if is_intrinsically_bad_line(line):
            continue

        if is_probable_repeated_boilerplate(
            line=line,
            page_frequency=repeated_line_counts[line],
            total_pages=total_pages,
            top_occurrences=line_top_counts[line],
            bottom_occurrences=line_bottom_counts[line],
        ):
            continue

        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_usable_page_text(text: str) -> bool:

    # Skip pages that still do not contain enough real content after cleaning.

    if not text:
        return False

    if len(text) < 300:
        return False

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if len(sentences) < 2:
        return False

    # Require some natural-language density
    stats = line_content_stats(text)
    if stats["alpha_ratio"] < 0.45:
        return False

    return True


def split_into_chunks(
    text: str,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        current_sentences = [sentences[i]]
        current_len = len(sentences[i])
        j = i + 1

        while j < n:
            next_sent = sentences[j]
            candidate_len = current_len + 1 + len(next_sent)

            if candidate_len > max_chars:
                break

            current_sentences.append(next_sent)
            current_len = candidate_len
            j += 1

        chunk_text = " ".join(current_sentences).strip()
        chunks.append(chunk_text)

        if j >= n:
            break

        if overlap_sentences > 0:
            i = max(j - overlap_sentences, i + 1)
        else:
            i = j

    return chunks


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

        pdf_page_lines: List[List[str]] = []

        # Step 1: extract lines page-by-page
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text()

            if not raw_text:
                logger.warning(
                    f"No text extracted from {pdf_path.name} page {page_number}")
                pdf_page_lines.append([])
                continue

            lines = raw_text.splitlines()
            pdf_page_lines.append(lines)

        total_pages = len(pdf_page_lines)

        # Step 2: collect repeated-line statistics across pages
        repeated_line_counts: Counter = Counter()
        line_top_counts: Counter = Counter()
        line_bottom_counts: Counter = Counter()

        for page_lines in pdf_page_lines:
            normalized_lines = [normalize_line(line) for line in page_lines]
            normalized_lines = [line for line in normalized_lines if line]

            # unique lines per page for page-frequency counts
            unique_lines = set(normalized_lines)
            repeated_line_counts.update(unique_lines)

            # track top/bottom positional occurrences
            top_lines = normalized_lines[:3]
            bottom_lines = normalized_lines[-3:] if len(
                normalized_lines) >= 3 else normalized_lines

            line_top_counts.update(set(top_lines))
            line_bottom_counts.update(set(bottom_lines))

        # Step 3: clean each page, then chunk it
        for page_number, page_lines in enumerate(pdf_page_lines, start=1):
            cleaned_txt = clean_page_lines(
                page_lines=page_lines,
                repeated_line_counts=repeated_line_counts,
                line_top_counts=line_top_counts,
                line_bottom_counts=line_bottom_counts,
                total_pages=total_pages,
            )

            if not cleaned_txt:
                logger.warning(
                    f"No usable cleaned text from {pdf_path.name} page {page_number}")
                continue

            if not is_usable_page_text(cleaned_txt):
                logger.debug(
                    f"Skipping low-value page text from {pdf_path.name} page {page_number}"
                )
                continue

            logger.debug(
                f"Extracted ~{len(cleaned_txt)} cleaned characters from "
                f"{pdf_path.name} page {page_number}"
            )

            page_chunks = split_into_chunks(
                cleaned_txt,
                max_chars=1000,
                overlap_sentences=1,
            )

            logger.debug(
                f"Created {len(page_chunks)} chunks from {pdf_path.name} page {page_number}"
            )

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


if __name__ == "__main__":
    chunks = ingest_pdfs()
    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print("=" * 80)
        print("ID:", chunk.id, "\n")
        print("Metadata:", chunk.metadata, "\n")
        print("Text:\n", chunk.text[:], "\n")
