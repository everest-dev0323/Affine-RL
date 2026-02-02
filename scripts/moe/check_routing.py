import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./your-merged-moe"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# This displays the 'routing' weights for the experts
# 0 = Gaming Expert, 1 = Logic Expert (depending on your config order)
def check_routing(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs, output_router_logits=True)
    
    # router_logits is a tuple (one for each layer)
    # Let's look at the routing decision for the last layer
    last_layer_logits = outputs.router_logits[-1] 
    probs = torch.nn.functional.softmax(last_layer_logits, dim=-1)
    
    print(f"Expert Weights for '{input_text}':")
    print(f" - Expert 1 (Gaming): {probs[0][-1][0]:.2f}")
    print(f" - Expert 2 (Logic): {probs[0][-1][1]:.2f}")

messages =  [
    {
        'role': 'system',
        'content': 'You are playing hex.\n\n# Game Rules\nHEX RULES:\nBoard: Diamond-shaped grid (5×5, 7×7, 9×9, or 11×11). Two players (Red and Blue).\nGoal: Connect your two opposite sides of the board with an unbroken chain of your stones.\n\nTurn: Place one stone of your color on any empty cell.\nRed (x) connects top-left to bottom-right sides.\nBlue (o) connects top-right to bottom-left sides.\n\nNo draws possible: Someone must win.\n\n\n# Output Format\nYou must respond with ONLY the action ID (a single number).\nDo NOT include descriptions or explanations.\n\nExamples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'},
    {
        'role': 'user',
        'content': 'Current State:\n. . . . . . . . . . . \n . . . . . . . . . . . \n  . . . . . . . . . . . \n   . . . . . . . . . . . \n    . . . . . . . . . . . \n     . . . . . . . . . . . \n      . . . . . . . . . . . \n       . . . . . . . . . . . \n        . . . . . . . . . . . \n         . . . . . . . . . . . \n          . . . . . . . . . . . \n\nYou are Player 0.\nLegal Actions:\n0 -> a1\n1 -> b1\n2 -> c1\n3 -> d1\n4 -> e1\n5 -> f1\n6 -> g1\n7 -> h1\n8 -> i1\n9 -> j1\n10 -> k1\n11 -> a2\n12 -> b2\n13 -> c2\n14 -> d2\n15 -> e2\n16 -> f2\n17 -> g2\n18 -> h2\n19 -> i2\n20 -> j2\n21 -> k2\n22 -> a3\n23 -> b3\n24 -> c3\n25 -> d3\n26 -> e3\n27 -> f3\n28 -> g3\n29 -> h3\n30 -> i3\n31 -> j3\n32 -> k3\n33 -> a4\n34 -> b4\n35 -> c4\n36 -> d4\n37 -> e4\n38 -> f4\n39 -> g4\n40 -> h4\n41 -> i4\n42 -> j4\n43 -> k4\n44 -> a5\n45 -> b5\n46 -> c5\n47 -> d5\n48 -> e5\n49 -> f5\n50 -> g5\n51 -> h5\n52 -> i5\n53 -> j5\n54 -> k5\n55 -> a6\n56 -> b6\n57 -> c6\n58 -> d6\n59 -> e6\n60 -> f6\n61 -> g6\n62 -> h6\n63 -> i6\n64 -> j6\n65 -> k6\n66 -> a7\n67 -> b7\n68 -> c7\n69 -> d7\n70 -> e7\n71 -> f7\n72 -> g7\n73 -> h7\n74 -> i7\n75 -> j7\n76 -> k7\n77 -> a8\n78 -> b8\n79 -> c8\n80 -> d8\n81 -> e8\n82 -> f8\n83 -> g8\n84 -> h8\n85 -> i8\n86 -> j8\n87 -> k8\n88 -> a9\n89 -> b9\n90 -> c9\n91 -> d9\n92 -> e9\n93 -> f9\n94 -> g9\n95 -> h9\n96 -> i9\n97 -> j9\n98 -> k9\n99 -> a10\n100 -> b10\n101 -> c10\n102 -> d10\n103 -> e10\n104 -> f10\n105 -> g10\n106 -> h10\n107 -> i10\n108 -> j10\n109 -> k10\n110 -> a11\n111 -> b11\n112 -> c11\n113 -> d11\n114 -> e11\n115 -> f11\n116 -> g11\n117 -> h11\n118 -> i11\n119 -> j11\n120 -> k11\n\nYour choice (ID only):'}
    ]

text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)

check_routing(text)