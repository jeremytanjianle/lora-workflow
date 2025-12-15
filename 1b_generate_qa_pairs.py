import json
from typing import List
from pydantic import BaseModel

from colorama import Fore
from loguru import logger
from litellm import completion

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def prompt_template(data: str, num_records: int = 5):

    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
    the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model. 

    For each of the {num_records} entries, generate one or two well-structured questions that reflect different aspects of the information in the chunk. 
    Ensure a mix of longer and shorter questions, with shorter ones typically containing 1-2 sentences and longer ones spanning up to 3-4 sentences. Each 
    Q&A pair should be concise yet informative, capturing key insights from the data.

    Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:

        "question": "Your question here...",
        "answer": "Your answer here..."

    Focus on creating clear, relevant, and varied questions that encourage the model to learn from diverse perspectives. Avoid any sensitive or biased 
    content, ensuring answers are accurate and neutral.

    Example:
    
        "question": "What is the primary purpose of this dataset?",
        "answer": "This dataset serves as training data for fine-tuning a language model."
    

    By following these guidelines, you'll contribute to a robust and effective dataset that enhances the model's performance."

    ---

    **Explanation:**

    - **Clarity and Specificity:** The revised prompt clearly defines the role of the assistant and the importance of the task, ensuring alignment with the 
    project goals.
    - **Quality Standards:** It emphasizes the need for well-formulated Q&A pairs, specifying the structure and content of each question and answer.
    - **Output Format:** An example JSON structure is provided to guide the format accurately.
    - **Constraints and Biases:** A note on avoiding sensitive or biased content ensures ethical considerations are met.
    - **Step-by-Step Guidance:** The prompt breaks down the task into manageable steps, making it easier for the assistant to follow.

    This approach ensures that the generated data is both high-quality and meets the specific requirements of the machine learning project.
    
    Data
    {data}
    """

def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )
    data = ""
    for x in stream:
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None:
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="")
            data += delta
    return json.loads(data)

if __name__ == "__main__":

    logger.info("Loading parsed PDF chunks")
    with open('./data/1a_parsed_pdf_chunks.json', 'r') as f:
        chunks_data = json.load(f)

    logger.info(f"Generating Q&A pairs for {len(chunks_data)} chunks")
    dataset = {}
    for chunk_id, chunk_info in chunks_data.items():
            print(Fore.YELLOW + f"\nChunk {chunk_id}:" + Fore.RESET)
            print(Fore.LIGHTMAGENTA_EX + f"Contextualized Text:\n{chunk_info['contextualized_text'][:300]}…" + Fore.RESET)

            data = llm_call(
                chunk_info['contextualized_text']
            )
            dataset[chunk_id] = {"generated": data["generated"], "context": chunk_info['contextualized_text']}

    logger.info(f"Saving dataset to tm1data.json")
    with open('./data/instruction.json', 'w') as f:
        json.dump(dataset, f, indent=2) 
