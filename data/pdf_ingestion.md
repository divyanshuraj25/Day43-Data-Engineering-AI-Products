# PDF Ingestion

PDF files are parsed page by page. Page boundaries are preserved as metadata where possible. Repeated page headers and footers should be removed only when confidently identified. Scanned PDFs may require OCR; a text parser alone cannot reliably extract text from image-only pages.
