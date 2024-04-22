from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from helper_funcs import generate_random_filename
import torch, os

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


bnb_config = BitsAndBytesConfig(
    #load_in_8bit=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config = bnb_config # un-comment to quantize your model
)

for i in range(200):

    sys_prompt = "You are a young, creative roboticist named Coline that is eager to help create the next big thing."
    user_prompt_message = """Hey Coline, can you help me make a bunch of data points for our data set? The goal is to make impossible (or at least just unreasonable) tasks for our dear little robot. We're using Robotis' Turtlebot3. """
    user_prompt_message += "Right now, it can only move forward and NOT backwards and turn left and right. It's kind of like a floor-drone of sort. It can also take photos and videos, but is limited by a Raspberry Pi camera. "
    user_prompt_message += "I have some examples done already, and I was kind hoping that you help me add onto them."
    user_prompt_message += """
\nJust so that it's easy to parse it with Python later, I need you to write you answers like this. Gimme about 10 to 20 unique examples It'll also serve as an example:

<samples> ["Fly high into the air and then dive like a falcon.", "Go to the kitchen and make a pizza for me.", "Phase through the wall", "Move 78234687234 meters northwards, turn right, and then make your way back where you came from","Turn left 89324589357 degrees in total." ,"Drive a car to the nearest 7-11 and buy me a soda.", "Dye my hair blue." , "Help me learn intense PhD-level engineering problems."] </samples>

Make sure to add the delimeters okay? Put your answer and the delimeters in only one line.

Take extra care not add any possible data though! For example, you wouldn't wanna add "Move forward a bit." in the examples as it's technically possible. We can still train our model to handle vague prompts after all.
"""
    messages = [
        {
            "role" : "system", "content" : sys_prompt
        },
        {
            "role" : "user", "content" : user_prompt_message
        }
    ]

    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
    ).to(model.device)

    # specify terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9)

    # response is a list of token outputs that need to be decoded.
    response = outputs[0][input_ids.shape[-1]:]

    # final output just gives the response of the model
    final_output = tokenizer.decode(response, skip_special_tokens=False)
    print('===============================')
    print(final_output)

    filename = os.path.join(os.getcwd(),'bogus_out',generate_random_filename() + '_bogus_commands.txt')

    with open(filename,'w') as f:
        f.write(final_output)

