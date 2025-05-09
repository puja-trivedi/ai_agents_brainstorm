import json
import os
import re
from datetime import datetime
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pydantic import Field
from langchain_core.utils import secret_from_env
from tqdm import tqdm
# Example per-1M-token pricing (USD) – adjust or extend as needed
MODEL_PRICING = {
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40}, 
    "gpt-4o-mini":     {"input": 0.15, "output": 0.60}, 
    "claude-3.7-sonnet": {"input": 3.0, "output": 15.0},
    "deepseek-chat-v3-0324": {"input": 0.0, "output": 0.0},
}

def compute_cost(input_tokens: int,
                 completion_tokens: int,
                 model: str) -> float:
    """
    Compute the estimated cost (in USD) for a single LLM call.

    Parameters:
    - input_tokens (int):   Number of prompt/input tokens.
    - completion_tokens (int): Number of completion/output tokens.
    - model (str):          Key in MODEL_PRICING indicating which model's rates to use.

    Returns:
    - float: Total cost in USD.

    Raises:
    - ValueError: If the model is not in MODEL_PRICING.
    """
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model '{model}'. Available models: {list(MODEL_PRICING)}")
    rates = MODEL_PRICING[model]
    cost_input  = input_tokens      / 1000000 * rates["input"]
    cost_output = completion_tokens / 1000000 * rates["output"]
    return cost_input + cost_output


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )
       
def parse_pdf(file_path, grobid_client):
    """
    Extract content from a PDF file using GROBID client.
    
    Args:
        file_path: Path to the PDF file
        grobid_client: GROBID client instance
        
    Returns:
        Extracted content from the PDF
    """
    def _concatenate_content(section):
        """
        Recursively concatenate the 'content' list into a single string
        and apply the same operation to any nested subsections.
        """
        if isinstance(section.get("content"), list):
            section["content"] = " ".join(section["content"])
        
        for subsection in section.get("subsections", []):
            _concatenate_content(subsection)

    xml_content = grobid_client.process_pdf(file_path)
    result = grobid_client.extract_content(xml_content)

    # for section in result.get('sections', []):
    #     _concatenate_content(section)

    # format output

    return result


def get_model_output(prompt, model, input_variables):
    """
    Get output from a language model using a given prompt and input text.
    
    Args:
        prompt: PromptTemplate object containing the prompt template
        model: Language model to use (e.g. llm_gpt_4o_mini, llm_deepseek_v3)
        input_variables: Input text to process
        
    Returns:
        Output from the model chain
    """
    model_chain = prompt | model
    return model_chain.invoke(input_variables)

def create_chain(llm, prompt):
    """
    Create an LLMChain with the given language model and prompt.
    
    Args:
        llm: Language model instance
        prompt: Prompt template
        
    Returns:
        LLMChain instance
    """
    return prompt | llm

def process_and_save_output(output, file_prefix="output", prompt=None):
    """
    Process LLM output to extract JSON, add model metadata, and save to file.
    
    Args:
        output: LLM output object containing content and response_metadata
        file_prefix: Prefix for the output filename (default: "extracted_entities")
    
    Returns:
        dict: Processed JSON object if successful, None otherwise
    """
    match = re.search(r"```(?:json)?\n(.*?)```", output.content, re.DOTALL)
    if match:
        json_str = match.group(1)
        json_obj = json.loads(json_str)
        # Add model name to JSON object
        json_obj["model_name"] = output.response_metadata["model_name"]
        if prompt:
            prompt_dict = {
            "template": prompt.template,
            "input_variables": prompt.input_variables
            }
            json_obj["prompt"] = prompt_dict
        print(json.dumps(json_obj, indent=2))
        # Save the JSON object to a file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = output.response_metadata["model_name"].replace('/', '_')
        filename = f'{file_prefix}_{model_name}_{timestamp}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_obj, f, indent=2)
        return json_obj
    return None


def get_entity_positions(raw_text: str, in_place_annotation: str):
    """
    Extracts annotated entities from in_place_annotation, finds their positions
    in raw_text, and returns a list of dictionaries with entity details.

    Matches both:
      (entity)[LABEL]
    and
      ((entity)[LABEL])
    """
    # Regex to capture optionally wrapped ((entity)[LABEL]) or plain (entity)[LABEL]
    pattern = re.compile(r'\(?\(([^)]+?)\)\[([^\]]+?)\]\)?')
    matches = pattern.findall(in_place_annotation)

    entities = []
    search_start = 0

    for entity, label in matches:
        start_index = raw_text.find(entity, search_start)
        if start_index == -1:
            entities.append({
                "entity": entity,
                "label": label
            })
        else:
            end_index = start_index + len(entity)
            entities.append({
                "entity": entity,
                "start_index": start_index,
                "end_index": end_index,
                "label": label
            })
            search_start = end_index

    return entities

#test positions 
def test_entity_positions(positions, raw_text):
    for entity in positions:
        start = entity.get("start_index")
        end = entity.get("end_index")
        if raw_text[start:end] != entity.get("entity"):
            print(f"Error: {raw_text[start:end]} != {entity.get('entity')}")
            return False
    return True

def run_chain_on_small_chunks(input_data, chain):
    """
    Runs chain on all 'content' and 'subsections' of 'sections' in the input dictionary.

    Parameters:
        input_data (dict): A dictionary with a 'sections' key, each section containing 'content' and 'subsections'.
        chain (Callable): A callable that takes a dict with 'neuroscience_text' and returns output.

    Returns:
        list: Alloutputs collected from the sections.
    """
    all_output = []
    sections = input_data.get('sections', [])

    for section in tqdm(sections, desc="Running NER on sections"):
        for chunk in tqdm(section.get('content', []), desc="Processing content", leave=False):
            output = chain.invoke({"neuroscience_text": chunk})
            all_output.append({"raw_text": chunk, "in_place_annotation": output})
        for chunk in tqdm(section.get('subsections', []), desc="Processing subsections", leave=False):
            output = chain.invoke({"neuroscience_text": chunk})
            all_output.append({"raw_text": chunk, "in_place_annotation": output})

    return all_output



def process_and_save_output_multiple(output, file_prefix="output", prompt=None):
    """
    Process LLM output(s) to extract JSON, add model metadata and prompt once,
    and save to a single JSON file.

    Args:
        output: A single LLM output or a list of LLM outputs (e.g., AIMessage) each with `content` and `response_metadata`.
        file_prefix: Prefix for the output filename.
        prompt: Optional PromptTemplate object with template and input_variables.

    Returns:
        dict: A dictionary containing metadata and all processed outputs.
    """
    if not isinstance(output, list):
        output = [output]

    processed_outputs = {}


    for index, item in enumerate(output):
        match = re.search(r"```(?:json)?\n(.*?)```", item.content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                json_obj = json.loads(json_str)
                processed_outputs[index+1] = json_obj
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON at index {index + 1}.")
                processed_outputs[index + 1] = {"error": "Failed to decode JSON",
                                                "response_metadata": item.response_metadata}
        else:
            print(f"Warning: No JSON block found in section {index + 1}.")
            processed_outputs[index + 1] = {"error": "No JSON block found",
                                            "response_metadata": item.response_metadata}

    if processed_outputs:
        model_name = output[0].response_metadata.get("model_name", "unknown")
        if prompt:
            prompt_metadata = {
                "template": prompt.template,
                "input_variables": prompt.input_variables
            }
        result = {
            "model_name": model_name,
            "prompt": prompt_metadata,
            "output": processed_outputs
        }


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_model_name = model_name.replace('/', '_')
        filename = f'{file_prefix}_{safe_model_name}_{timestamp}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(f"Saved {len(processed_outputs)} outputs to {filename}")
        return result
    else:
        print("No valid outputs to save.")
        return None

def batch_extract_and_format(records, model_name, save_to_file=True, file_prefix='output', prompt=None):
    """
    Process a batch of raw + in-place annotated texts, extracting entities
    only when LLM finished normally (finish_reason == 'stop'). Otherwise,
    store the raw response_metadata.
    Returns a dict of the form:
    {
      1: {
         "raw_text": ...,
         "in_place_annotation": ...,
         # either "entities": [...] or "response_metadata": {...}
      },
      2: { ... },
      ...
    }
    """
    results = {}
    input_tokens = 0
    completion_tokens = 0
    for idx, record in enumerate(records, start=1):
        raw = record["raw_text"]
        msg = record["in_place_annotation"]
        # get content string and metadata dict
        ann = msg.content
        metadata = msg.response_metadata
        input_tokens += msg.usage_metadata.get("input_tokens", 0)
        completion_tokens += msg.usage_metadata.get("output_tokens", 0)
        #print(f'input_tokens: {input_tokens}, completion_tokens: {completion_tokens}')
        # check finish reason
        if metadata.get("finish_reason") == "stop":
            # normal completion → extract entities
            entities = get_entity_positions(raw, ann)
            results[idx] = {
                "raw_text": raw,
                "in_place_annotation": ann,
                "entities": entities
            }
        else:
            # aborted or truncated → capture metadata for debugging
            results[idx] = {
                "raw_text": raw,
                "in_place_annotation": ann,
                "response_metadata": metadata
            }
    # compute costs
    total_cost = compute_cost(input_tokens, completion_tokens, model_name)
    #print(f"Total cost for {model_name}: ${total_cost:.2f}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if prompt:
        prompt_metadata = {
            "template": prompt.template,
            "input_variables": prompt.input_variables
        }
    else:
        prompt_metadata = None
    if save_to_file:
        # save results to a JSON file
        file_name = f"{file_prefix}_{model_name}_{timestamp}.json"
        #create a file
        with open(file_name, "w") as f:
            json.dump({"model_name": model_name, "prompt": prompt_metadata, "total_cost": total_cost, "results": results}, f, indent=2)
            print(f"Results saved to {file_name}")

    else:
        print({"model_name": model_name, "prompt": prompt_metadata, "total_cost": total_cost, "results": results})
