# this is taken from the Llama 3 Huggingface repo.
# for a chat-like interface, the Instruct model is far superior than its base model.

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# OPTIONAL: Load the model in 8-bit/4-bit to increase inference speed and lower VRAM usage at the expense of some 
# performance loss

# When using 8-bit, comment-out the 4-bit arguments.
bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# load the model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config = bnb_config # un-comment to quantize your model
)

# put the messages in a message list to format the data
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# apply chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

print(tokenizer.decode(input_ids))

# specify terminators
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# generate the output
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# response is a list of token outputs that need to be decoded.
response = outputs[0][input_ids.shape[-1]:]

# final output just gives the response of the model
final_output = tokenizer.decode(response, skip_special_tokens=False)
print(final_output)