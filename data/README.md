# Data

No documents ship with this repo. Put a handful of PDFs in `data/pdfs/` yourself.

Suggested source: Florida Housing Finance Corporation (FHFC), which publishes Request for
Applications (RFA) material, application documents and funding or scoring reports on its website
(floridahousing.org). Pick 5 to 10 documents with a mix of layouts: at least one text-heavy
application, one table-heavy report and, if you want to exercise OCR, one scanned page.

Before you use any file:

1. Confirm it is publicly posted by the agency and not marked confidential.
2. Note the URL and download date next to the gold label (`labelled_on` in the gold JSON).
3. Do not commit the PDFs. `.gitignore` already excludes `data/pdfs/*`.

Field labels differ between programmes and years. Expect to add label synonyms to
`src/pdf_extractor/schemas.py` after looking at your documents; that is part of the work, not a bug.
