from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "demonwizard0/affine-new-3"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {
        "role": "user",
        "content": 'You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".\nEvery round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look-up the game inventory by "inventory", or "craft" (target) using any of the crafting commands.\nYour output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"\n\nReminder: \n1. Always specify the quantity when using "get" and "craft" commands. - Example of get: get 1 lapis lazuli - Example1 of craft: craft 1 blue dye using 1 lapis lazuli - Example2 of craft: craft 1 golden carrot using 8 gold nugget, 1 carrot\n2. When using "get" command, do not specify whether the item comes from the inventory or the environment.\n3. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.\n\n',
        "reasoning_content": None,
    },
    {
        "role": "assistant",
        "content": "OK. I'll follow your instructions and try my best to solve the task.",
        "reasoning_content": None,
    },
    {
        "role": "user",
        "content": "Crafting commands:\ncraft 3 acacia sign using 6 acacia planks, 1 stick\ncraft 1 oak pressure plate using 2 oak planks\ncraft 4 warped planks using 1 warped stems\ncraft 4 spruce planks using 1 spruce logs\ncraft 4 jungle planks using 1 jungle logs\ncraft 6 scaffolding using 1 string, 6 bamboo\ncraft 4 birch planks using 1 birch logs\ncraft 1 note block using 8 planks, 1 redstone\ncraft 1 lime banner using 6 lime wool, 1 stick\ncraft 3 crimson sign using 6 crimson planks, 1 stick\ncraft 1 green banner using 6 green wool, 1 stick\ncraft 4 acacia planks using 1 acacia logs\ncraft 1 oak fence gate using 4 stick, 2 oak planks\ncraft 1 magenta bed using 3 magenta wool, 3 planks\ncraft 4 crimson planks using 1 crimson stems\ncraft 1 cyan bed using 3 cyan wool, 3 planks\ncraft 1 stick using 2 bamboo\ncraft 1 wooden axe using 2 stick, 3 planks\ncraft 4 dark oak planks using 1 dark oak logs\ncraft 4 stick using 2 planks\ncraft 4 oak planks using 1 oak logs\n\nGoal: craft oak fence gate.",
        "reasoning_content": None,
    },
    {
        "role": "assistant",
        "content": "Thought:\nTo craft an oak fence gate I need 4 sticks and 2 oak planks. I can obtain oak planks by crafting them from oak logs, and I can craft sticks from planks. First I should get an oak log.\n\nAction:\nget 1 oak logs",
        # "reasoning_content": 'We need to craft oak fence gate, which requires 4 stick, 2 oak planks.\n\nWe have to get ingredients. We can craft sticks from planks, or get bamboo to craft sticks, etc. We need oak planks: can craft 4 oak planks using 1 oak logs. So need oak logs (quantity? need 2 oak planks, but recipe yields 4 per log, so 1 oak log is enough). Need 4 sticks: can craft 4 stick using 2 planks (any planks). So need 2 planks (any type). Could use oak planks we already have. So steps:\n\n1. get 1 oak logs (to make oak planks)\n2. craft 4 oak planks using 1 oak logs (we\'ll have 4 oak planks)\n3. get 2 planks? Actually we already have oak planks, we need 2 planks for sticks. Use 2 of the oak planks, leaving 2 oak planks for gate. So we need to craft sticks: craft 4 stick using 2 planks (oak planks). That consumes 2 oak planks, leaving 2 oak planks.\n4. craft oak fence gate using 4 stick, 2 oak planks.\n\nWe need 4 sticks; craft 4 stick yields 4 sticks, good.\n\nThus actions sequence.\n\nWe must specify quantities.\n\nLet\'s output thought then action. Probably need multiple actions separate lines? The format says "Action:\\nyour next action". Probably one action per turn. But can we do multiple? Probably each turn one action. The prompt: "Every round I will give you an observation, you have to respond an action". So we need only one action now. The observation is the list and goal. So we need to decide next action: get 1 oak logs. Or maybe we can combine? Usually you start by getting needed items. So action: get 1 oak logs.',
    },
    {"role": "user", "content": "Got 1 oak logs", "reasoning_content": None},
    {
        "role": "assistant",
        "content": "Thought:\nI need oak planks and sticks to craft the oak fence gate. I'll craft oak planks from the oak log, then craft sticks from some of those planks, and finally craft the fence gate.\n\nAction:\ncraft 4 oak planks using 1 oak logs\n\nAction:\ncraft 4 stick using 2 oak planks\n\nAction:\ncraft 1 oak fence gate using 4 stick, 2 oak planks",
        "reasoning_content": "We have oak log. Need 2 oak planks. Craft 4 oak planks from 1 oak log, then we will have 4 oak planks, enough. Need sticks: 4 sticks. We can craft sticks from planks: craft 4 stick using 2 planks. Each craft yields 4 sticks from 2 planks. So need 2 planks for sticks. We have oak planks (4). Use 2 for sticks, leaving 2 oak planks for fence gate. So steps: craft oak planks, craft sticks, then craft fence gate.\n\nWe must specify quantity: craft 4 oak planks using 1 oak logs. Then craft 4 stick using 2 oak planks. Then craft 1 oak fence gate using 4 stick, 2 oak planks. Ensure we have enough planks left: after stick craft we used 2 planks, leaving 2 planks for fence gate. Good.\n\nProceed.",
    },
    {
        "role": "user",
        "content": "Error: Only one 'Action' is allowed per response. Please adjust your response.",
        "reasoning_content": None,
    },
    # {
    #     "role": "assistant",
    #     "content": "Thought:\nI have 1 oak log. To eventually craft the oak fence gate I need oak planks and sticks. First I will convert the oak log into oak planks.\n\nAction:\ncraft 4 oak planks using 1 oak logs",
    #     "reasoning_content": None,
    # },
    # {
    #     "role": "user",
    #     "content": "Crafted 4 minecraft:oak_planks",
    #     "reasoning_content": None,
    # },
    # {
    #     "role": "assistant",
    #     "content": "Thought:\nI have 4 oak planks. I need 4 sticks for the gate, which can be crafted from 2 oak planks. I will craft the sticks now.\n\nAction:\ncraft 4 stick using 2 oak planks",
    #     "reasoning_content": "We now have 4 oak planks. Need 2 oak planks for gate, and need 4 sticks. Sticks can be crafted from 2 planks each (craft 4 stick using 2 planks). That yields 4 sticks from 2 planks. So we need total planks: 2 for gate + 2 for sticks = 4 planks, which we have. So we can craft sticks then gate.\n\nWe must output one action per response. So first craft sticks.\n\n",
    # },
    # {"role": "user", "content": "Crafted 4 minecraft:stick", "reasoning_content": None},
    # {
    #     "role": "assistant",
    #     "content": "Thought:\nI have at least 2 oak planks and 4 sticks, which are the required ingredients for the oak fence gate.\n\nAction:\ncraft 1 oak fence gate using 4 stick, 2 oak planks",
    #     "reasoning_content": "We have 4 oak planks (original) and 4 sticks. Need 2 oak planks and 4 sticks to craft oak fence gate. We have enough. So next action craft oak fence gate.\n\n",
    # },
    # {
    #     "role": "user",
    #     "content": "Crafted 1 minecraft:oak_fence_gate",
    #     "reasoning_content": None,
    # },
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

print("#" * 80)
print(text)
print("#" * 80)

try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0
    
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)    
