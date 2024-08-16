from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import re
import sys

app = Flask(__name__)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# adapter_path = "/app/results_v3.2/final_checkpoint"
adapter_path = "/app/results_v4.1/final_checkpoint"

# model = AutoModelForCausalLM.from_pretrained(
#     adapter_path,
#     torch_dtype=torch.bfloat16,
#     load_in_4bit=True,
# )

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)


tokenizer = AutoTokenizer.from_pretrained(adapter_path)

def extract_plan(output):
    actions_with_args = re.findall(r'\b(GO_TO|TAKE|PUT_IN|TILT|TURN|GO|SAY|SEARCH_VIEW|DESCRIBE_VIEW|SEARCH_DATA_BASE|QUESTION_VIEW|LISTEN|THOUGHT)\((.*?)\)', output)
    actions_without_args = re.findall(r'\b(GET_UP_AFTER_FALL|JUMP_TURN|DANCE|SIT|UP|FOLLOW|GO_USER|GIVE_TO_USER|FINISH|WAIT)\b', output)

    actions = []
    actions.extend(actions_with_args)
    actions.extend(actions_without_args)
    
    if actions:
        if len(actions[0]) == 2:
            action, args = actions[0]
            return action + "(" + args + ")"
        else:
            action = actions[0]
            return action
    return ""

def generate_step(prompt, max_tokens=1000):
    
    print("I got prompt: ", prompt)
   
    generated_tokens = []

    decoded_output = ""
    
    for _ in range(max_tokens):
        inputs = tokenizer.encode(prompt + decoded_output, return_tensors="pt").to(DEV)
        
        num_per_time=5

        generate_kwargs = dict(
            input_ids=inputs,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_new_tokens=num_per_time,
            repetition_penalty=1.3
        )
        outputs = model.generate(**generate_kwargs)
        generated_tokens.extend(outputs[0][-num_per_time:].tolist())
        
        decoded_output = tokenizer.decode(generated_tokens)
        
        # sys.stdout.write("\r" + decoded_output)
        # sys.stdout.flush()
        
        next_step = extract_plan(decoded_output)

        if next_step:
            break

    sys.stdout.write("\n")  # Переход на новую строку после завершения цикла
    sys.stdout.flush()
    
    print("I returne next step: ", next_step)

    return next_step

@app.route('/step_generation', methods=['POST'])
def generate_step_route():
    data = request.json
    prompt = data['prompt']
        
    next_step = generate_step(prompt)
    
    if "FINISH" in next_step:
        response = {
            'next_step': next_step,
            'message': 'Plan is finished!'
        }
    else:
        response = {
            'next_step': next_step,
            'message': 'Next step generated successfully!'
        }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7778)




