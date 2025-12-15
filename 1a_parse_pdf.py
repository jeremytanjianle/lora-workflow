import json
from colorama import Fore
from loguru import logger
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

if __name__ == "__main__":

    logger.info("Converting PDF document")
    converter = DocumentConverter()
    doc = converter.convert("tm1_dg_dvlpr-10pages.pdf").document

    logger.info("Chunking PDF document")
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    logger.info("Building contextualized chunks")
    chunks_data = {}
    for i, chunk in enumerate(chunks):
            print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}…" + Fore.RESET)
            enriched_text = chunker.contextualize(chunk=chunk)
            print(Fore.LIGHTMAGENTA_EX + f"Contextualized Text:\n{enriched_text[:300]}…" + Fore.RESET)

            chunks_data[i] = {
                "raw_text": chunk.text,
                "contextualized_text": enriched_text
            }

    logger.info(f"Saving {len(chunks_data)} chunks to parsed_pdf_chunks.json")
    with open('./data/1a_parsed_pdf_chunks.json', 'w') as f:
        json.dump(chunks_data, f, indent=2) 
