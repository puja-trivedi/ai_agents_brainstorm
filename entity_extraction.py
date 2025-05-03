import json
import os
import re
from datetime import datetime
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pydantic import Field
from langchain_core.utils import secret_from_env


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

    for section in result.get('sections', []):
        _concatenate_content(section)

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
