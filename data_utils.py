import os
import json

def load_gsm8k_data(data_path):
    data = {"question":[], "answer":[]}
    with open(data_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = json.loads(line)
            data["question"].append(line["question"])
            data["answer"].append(line["answer"].replace("####", "FINAL ANSWER:"))
    return data


def load_synthetic_data(dataset_file):
    data = {"question":[], "answer":[], "original_answer": []}
    with open(dataset_file, "r") as f:
        for line in f:
            line = json.loads(line)
            data["question"].append(line["question"])
            data["answer"].append(line["prediction"])
            data["original_answer"].append(line["original_answer"])
    return data

def load_prompt(prompt_path):
      with open(prompt_path, "r", encoding = "utf-8") as f:
          prompt = f.read()
      return prompt

class DatasetProcessor:
    def __init__(self,
                 tokenizer,
                 training_prompt_path):
        self.tokenizer = tokenizer
        self.training_prompt = load_prompt(training_prompt_path)

    
    def training_preprocessor_function(self, examples):
            inputs = [(self.training_prompt.format(question = examples["question"][idx]), examples["answer"][idx] + self.tokenizer.eos_token) for idx in range(len(examples["question"]))]
            #outputs = [examples["answer"][idx] for idx in range(len(examples["question"]))]
            model_inputs = self.tokenizer(inputs, padding="max_length",
                                           max_length=384, truncation=True,
                                           return_token_type_ids=True)
            model_inputs["original_queston"] = [self.training_prompt.format(question = examples["question"][idx]) for idx in range(len(examples["question"]))]
            model_inputs["original_answer"] = examples["answer"]
            tokenized_questions = self.tokenizer(model_inputs["original_queston"], padding="max_length", max_length=384)
            model_inputs["tokenized_question_ids"] = tokenized_questions["input_ids"]
            model_inputs["tokenized_question_attentions"] = tokenized_questions["attention_mask"]



            #labels = model.tokenizer(text_target=examples["answer"], max_length=512, truncation=True)

            #model_inputs["labels"] = model_inputs["input_ids"]
            return model_inputs
