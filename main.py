from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from pydantic import BaseModel

tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")


class Item(BaseModel):
    instruction: str
    input_text: str

# Init model, transforms


def generate_prompt(instruction, input_text=None):
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input_text that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)



def predict(item, run_id, logger):
    item = Item(**item)
    instruction = item.instruction
    input_text = item.instruction
    prompt = generate_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    output = ""
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
    return {"response": output.split("### Response:")[1].strip()}