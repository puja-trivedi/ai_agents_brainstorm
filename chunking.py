'''
Chunking approach for different tasks:
Retrieval/Augmented Generation → Embedding‑ or topic‑model‑based chunking
Summarization → Discourse‑structure or cue‑phrase informed segments
Annotation/NER → Smaller units like sentences or semantic‑role spans
'''

import json
# section level segmentation
def section_tokenizer(text:dict, save_to_file:bool=False, file_name:str="section_chunks.json"):
    """
    Segments a given text into sections and optionally saves the result to a file.

    Args:
        text (dict): A dictionary containing the text to be segmented. 
                        It should have a "sections" key, which is a list of sections, 
                        where each section is a dictionary with "heading" and "content" keys.
        save_to_file (bool, optional): If True, saves the segmented sections to a file. Defaults to False.
        file_name (str, optional): The name of the file to save the segmented sections. Defaults to "section_chunks.json".

    Returns:
        dict: A dictionary containing:
                - "metadata": The metadata from the input text (if any).
                - "chunks": A list of dictionaries, each representing a section with:
                    - "id": The section index as a string.
                    - "section_name": The heading of the section.
                    - "text": The concatenated content of the section.
    """
    all_chunks = []
    sections = text.get("sections", [])
    section_index = 1
    for section in sections:
        section_name = section.get("heading", "")
        content = section.get("content", [])
        chunk_id = f"{section_index}"
        chunk = {
            "id": chunk_id,
            "section_name": section_name,
            "text": "\n".join(content)
        }
        all_chunks.append(chunk)
        section_index += 1
    output = {"metadata": text.get("metadata", {}), "chunks": all_chunks}
    if save_to_file:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"Saved sentence chunks to {file_name}")
    return output

# paragraph level segmentation
def paragraph_tokenizer(text:dict, save_to_file:bool=False, file_name:str="paragraph_chunks.json"):
    """
    Segments the input text into paragraphs and organizes them into chunks with metadata.

    Args:
        text (dict): A dictionary containing the text to be segmented. 
        save_to_file (bool, optional): If True, saves the resulting chunks to a JSON file. Defaults to False.
        file_name (str, optional): The name of the file to save the chunks if `save_to_file` is True. Defaults to "paragraph_chunks.json".

    Returns:
        dict: A dictionary containing:
            - "metadata": The metadata from the input text (if any).
            - "chunks": A list of dictionaries, where each dictionary represents a paragraph chunk with the following keys:
                - "id": A unique identifier for the chunk in the format "section_index_paragraph_index".
                - "section_name": The name of the section the paragraph belongs to.
                - "text": The paragraph text.
    """
    all_chunks = []
    sections = text.get("sections", [])
    section_index = 1
    for section in sections:
        section_name = section.get("heading", "")
        content = section.get("content", [])
        paragraph_index = 1 
        for paragraph in content:
            chunk_id = f"{section_index}_{paragraph_index}"
            chunk = {
                "id": chunk_id,
                "section_name": section_name,
                "text": paragraph
            }
            all_chunks.append(chunk)
            paragraph_index += 1
        section_index += 1
    output = {"metadata": text.get("metadata", {}), "chunks": all_chunks}
    if save_to_file:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"Saved sentence chunks to {file_name}")
    return output

# sentence level segmentation
def sentence_tokenizer(text:dict, save_to_file:bool=False, file_name:str="sentence_chunks.json"):
    """
    Segments the input text into sentences and organizes them into chunks with unique IDs.

    Args:
        text (dict): A dictionary containing the text to be segmented. 
        save_to_file (bool, optional): If True, saves the resulting chunks to a JSON file. Defaults to False.
        file_name (str, optional): The name of the file to save the chunks if `save_to_file` is True. Defaults to "sentence_chunks.json".

    Returns:
        dict: A dictionary containing:
                - "metadata": The metadata from the input text.
                - "chunks": A list of sentence chunks, where each chunk is a dictionary with:
                    - "id": A unique identifier for the chunk in the format "sectionIndex_paragraphIndex_sentenceIndex".
                    - "section_name": The name of the section the sentence belongs to.
                    - "text": The sentence text.
    """
    all_chunks = []
    sections = text.get("sections", [])
    section_index = 1
    for section in sections:
        section_name = section.get("heading", "")
        content = section.get("content", [])
        paragraph_index = 1 
        for paragraph in content:
            sentences = paragraph.split(". ")
            sentence_index = 1
            for sentence in sentences:
                chunk_id = f"{section_index}_{paragraph_index}_{sentence_index}"
                chunk = {
                    "id": chunk_id,
                    "section_name": section_name,
                    "text": sentence
                }
                all_chunks.append(chunk)
                sentence_index += 1
            paragraph_index += 1
        section_index += 1
    output = {"metadata": text.get("metadata", {}), "chunks": all_chunks}
    if save_to_file:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"Saved sentence chunks to {file_name}")
    return output