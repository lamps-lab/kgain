import sys
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import precision_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='Finetune Qwen2')
parser.add_argument('directory', help="directory to save the predictions to")
parser.add_argument('model', help="Either model_id or directory where the weights of the model is")
args = parser.parse_args()
model_id = args.model
path = args.directory

with open("../data/test_dataset.json", "r") as f:
    test_samples = json.load(f)

print(f"number of test samples: {len(test_samples)}")
#exit()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
new_vocab_size = len(tokenizer)

# Update config to match the new vocab size
config = AutoConfig.from_pretrained(model_id)
config.vocab_size = new_vocab_size

model = AutoModelForCausalLM.from_pretrained(model_id, ignore_mismatched_sizes=True)
#model.config.vocab_size = len(tokenizer)
#model.resize_token_embeddings(len(tokenizer))


model.eval()
if torch.cuda.is_available():
    model.to("cuda")

def generate_answer(sample):
    prompt = (
        "Below is a multiple choice question based on the following content. "
        "Answer the question by picking one of the options, and provide a brief explanation of your choice.\n"
        "Output in JSON format with keys: \"article_id\", \"question\", \"answer\", \"explanation\".\n"
        "---\n"
        f"Article ID: {sample['article_id']}\n"
        f"Content: {sample['content']}\n"
        f"Question: {sample['question']}\n"
        "Only give me the letter of the option (not the text of the option) as the value for the answer key!\n For example, for T/F if statement is true give A, if you think False give B.\n"
        "Output only in JSON format with keys: \"article_id\", \"question\", \"answer\", \"explanation\"."
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,  # allow up to 256 new tokens regardless of input length
        do_sample=False,
        #pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"output: {output_text}\n\n")
    return output_text
def parse_multiple_json(text):
    """
    Extracts and parses all JSON objects from the given text.
    
    The algorithm scans character-by-character, tracking open and closed braces.
    When the brace count returns to 0, a candidate JSON substring is extracted.
    
    Equations:
        Let c(i) be the unmatched '{' count up to index i:
        \[
          c(i) =
            \begin{cases}
              c(i-1) + 1, & \text{if text}[i] = \{ \\
              c(i-1) - 1, & \text{if text}[i] = \} \\
              c(i-1), & \text{otherwise}
            \end{cases}
        \]
        A candidate JSON is found when c(i) = 0.
    """
    json_objects = []
    brace_count = 0
    start_index = None

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_index = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_index is not None:
                candidate = text[start_index:i+1]
                try:
                    obj = json.loads(candidate)
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    # Optionally add fixes for common JSON errors here.
                    pass
                start_index = None

    return json_objects
def parse_generated_json(text):
    """
    Attempt to extract a JSON object from the generated text.
    If the output is not valid JSON, extract the substring between
    the first '{' and the last '}'.
    """
    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                return None
        return None
def parse_output(text):
    regex_pattern = r"Answer:\s*(?P<answer>[A-Z])\s*(?:\n|\r\n|$).*?Explanation:\s*(?P<explanation>.+)"
    match = re.search(regex_pattern, text, re.DOTALL)
    if match:
        return [{
            "answer": match.group("answer"),
            "explanation": match.group("explanation").strip()
    }]
    return []

# -----------------------------------------------
# 4. RUN TESTING: Generate predictions and compute metrics
# -----------------------------------------------
predictions = []
ground_truths = []

for sample in test_samples:
    output_text = generate_answer(sample)
    parsed_output = parse_multiple_json(output_text)
    
    if parsed_output is None:
        parsed = parse_output(output_text)
        if len(parsed) == 0:
            continue
            pred = {
            "article_id": sample["article_id"],
            "content": sample["content"],
            "question": sample["question"],
            "answer": "",
            "explanation": ""
            }
        else:
            pred = {
            "article_id": sample["article_id"],
            "content": sample["content"],
            "question": sample["question"],
            "answer": parsed['answer'],
            "explanation": parsed['explanation']
            }
    else:
        pred = parsed_output
    predictions.append(pred)
    # The ground truth answer is available in the test sample.
    ground_truths.append(sample["answer"])

file = f"{path}test_predictions.json"
with open(file, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"predictions saved to {file}")
# -----------------------------------------------
# 5. CALCULATE EVALUATION METRICS
# -----------------------------------------------
pred_answers = []
for pred in predictions:
    # If pred is a list, take the first element (if available)
    if isinstance(pred, list):
        pred_dict = pred[0] if len(pred) > 0 and isinstance(pred[0], dict) else {}
    else:
        pred_dict = pred
    pred_answers.append(pred_dict.get("answer", ""))

# Now compute metrics using pred_answers
precision = precision_score(ground_truths, pred_answers, average='macro', zero_division=0)
recall = recall_score(ground_truths, pred_answers, average='macro', zero_division=0)
f1 = f1_score(ground_truths, pred_answers, average='macro', zero_division=0)

print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1 Score:  {f1}")

exit()
pred_answers = [pred.get("answer", "") for pred in predictions]

precision = precision_score(ground_truths, pred_answers, average='macro', zero_division=0)
recall = recall_score(ground_truths, pred_answers, average='macro', zero_division=0)
f1 = f1_score(ground_truths, pred_answers, average='macro', zero_division=0)

print("Evaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


