"""
PDF Processor v3
----------------
Features:
1. Fast PyMuPDF extraction
2. pdfplumber fallback
3. Advanced text cleanup
4. Broken-word reconstruction
5. OCR spacing fixes
6. Better RAG chunk quality
7. Cleaner semantic retrieval
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz
import pdfplumber

logger = logging.getLogger(__name__)


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class PDFPage:
    page_num: int
    text: str
    source: str


@dataclass
class PDFDocument:
    filename: str
    filepath: str
    pages: List[PDFPage] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            p.text for p in self.pages
            if p.text.strip()
        )


# =========================================================
# PDF PROCESSOR
# =========================================================

class PDFProcessor:

    MIN_CHARS = 50

    # =====================================================
    # MAIN EXTRACTION
    # =====================================================

    def extract(
        self,
        filepath: str,
        filename: str
    ) -> PDFDocument:

        logger.info("Extracting PDF: %s from %s", filename, filepath)

        doc = PDFDocument(
            filename=filename,
            filepath=filepath
        )

        if not Path(filepath).exists():

            raise FileNotFoundError(
                f"File not found: {filepath}"
            )

        # PRIMARY EXTRACTION
        try:

            doc.pages = self._extract_pymupdf(
                filepath,
                filename
            )
            logger.info("PyMuPDF extracted %d pages from %s", len(doc.pages), filename)

        except Exception as exc:

            logger.warning(
                f"PyMuPDF failed ({exc}), using pdfplumber"
            )

            doc.pages = self._extract_pdfplumber(
                filepath,
                filename
            )
            logger.info("pdfplumber extracted %d pages from %s", len(doc.pages), filename)

            logger.info(
                "PDF extraction complete: %s — %d pages, %d chars",
                filename, doc.total_pages, len(doc.full_text),
            )
            return doc

        # BLANK PAGE FALLBACK
        blank_pages = [
            p for p in doc.pages
            if len(p.text.strip()) < self.MIN_CHARS
        ]

        if blank_pages:

            logger.info(
                f"{len(blank_pages)} blank pages — retrying with pdfplumber"
            )

            plumber_pages = {
                p.page_num: p
                for p in self._extract_pdfplumber(
                    filepath,
                    filename
                )
            }

            for page in doc.pages:

                if (
                    len(page.text.strip()) < self.MIN_CHARS
                    and page.page_num in plumber_pages
                ):

                    page.text = plumber_pages[
                        page.page_num
                    ].text

        logger.info(
            f"Extracted {doc.total_pages} pages from '{filename}'"
        )

        return doc

    # =====================================================
    # PYMUPDF
    # =====================================================

    def _extract_pymupdf(
        self,
        filepath: str,
        filename: str
    ) -> List[PDFPage]:

        pages = []

        with fitz.open(filepath) as pdf:

            for i, page in enumerate(pdf, start=1):

                # BETTER TEXT EXTRACTION
                text = page.get_text("text")

                # CLEAN TEXT
                text = self._clean(text)

                pages.append(
                    PDFPage(
                        page_num=i,
                        text=text,
                        source=filename,
                    )
                )

        return pages

    # =====================================================
    # PDFPLUMBER
    # =====================================================

    def _extract_pdfplumber(
        self,
        filepath: str,
        filename: str
    ) -> List[PDFPage]:

        pages = []

        with pdfplumber.open(filepath) as pdf:

            for i, page in enumerate(pdf.pages, start=1):

                text = page.extract_text() or ""

                text = self._clean(text)

                pages.append(
                    PDFPage(
                        page_num=i,
                        text=text,
                        source=filename,
                    )
                )

        return pages

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    @staticmethod
    def _clean(text: str) -> str:

        if not text:
            return ""

        # REMOVE NULL BYTES
        text = text.replace("\x00", "")

        # NORMALIZE LINE BREAKS
        text = re.sub(r"\r", "\n", text)

        # REMOVE TABS
        text = re.sub(r"\t", " ", text)

        # REMOVE EXTRA NEWLINES
        text = re.sub(r"\n{2,}", "\n", text)

        # REMOVE MULTIPLE SPACES
        text = re.sub(r"[ ]{2,}", " ", text)

        # FIX SPLIT YEARS
        # 194 7 -> 1947
        text = re.sub(
            r'(\d{3})\s+(\d)',
            r'\1\2',
            text
        )

        # FIX HYPHEN SPACING
        # sub - campus -> sub-campus
        text = re.sub(
            r'\s*-\s*',
            '-',
            text
        )

        # FIX COMMON OCR SPLITS
        replacements = {
            "Sind h": "Sindh",
            "Jam sh oro": "Jamshoro",
            "All ama": "Allama",
            "K azi": "Kazi",
            "Ind us": "Indus",
            "Mo en-jo-D aro": "Moen-jo-Daro",
            "de part ments": "departments",
            "cent res": "centres",
            "sub-c amp uses": "sub-campuses",
        }

        for wrong, correct in replacements.items():

            text = text.replace(
                wrong,
                correct
            )

        # FIX SMALL WORD BREAKS
        # Example:
        # univer sity -> university
        text = re.sub(
            r'(\b\w{2,6})\s(?=\w{1,4}\b)',
            r'\1',
            text
        )

        # REMOVE SPACE BEFORE PUNCTUATION
        text = re.sub(
            r'\s+([.,!?;:])',
            r'\1',
            text
        )

        # FINAL CLEANUP
        text = re.sub(
            r'\s+',
            ' ',
            text
        )

        return text.strip()