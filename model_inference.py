from vllm import LLM, SamplingParams
from data_utils import load_gsm8k_data, load_prompt
from evaluator import Evaluator
import json
import os
from datasets import Dataset
import torch
import time
from pynvml import *
import math
from transformers import AutoTokenizer
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)

class Generator:
  def __init__(self,
              model_name : str,
              mode : str = "eval-top1"):
      self.generation_prompt = load_prompt("sft_generation.txt")
      self.eval_prompt = load_prompt("sft_eval.txt")

      self.model = LLM(model=model_name,
          trust_remote_code=True
          )
      self.tokenizer = AutoTokenizer.from_pretrained(
          model_name, device_map = "auto",
          add_eos_token=True,
        )
      self.token_mapping = {"TRUE": self.tokenizer.convert_tokens_to_ids("True"),
                            "FALSE": self.tokenizer.convert_tokens_to_ids("False")}
      self.mode = mode
      self.batch_size = 16 # specify generation batch size
      if mode == "eval-top1":
            self.batch_size = 16 # specify generation batch size
            self.sampling_params = SamplingParams(max_tokens=384,
                                    top_k = 40,
                                    temperature = 0.3,
                                    n = 1,
                                    best_of = 5)
      if mode in  ["eval-majority_vote", 
                   "eval-conf_top1", 
                   "eval-conf_classifier_maj_voting", 
                   "data_gen-top_1_confidence_threshold",
                   "data_gen-majority_vote",
                   "data_gen-conf_classifier_maj_voting"]:
            self.batch_size = 8 # specify generation batch size
            self.sampling_params = SamplingParams(max_tokens=384,
                                    top_k = 40,
                                    temperature = 0.7,
                                    n = 16)
            self.eval_params = SamplingParams(max_tokens=384,
                                    top_k = 40,
                                    temperature = 0,
                                    n = 1,
                                    logprobs = 4)
      elif mode == "eval-top16":
            self.batch_size = 8 # specify generation batch size
            self.sampling_params = SamplingParams(max_tokens=384,
                                    top_k = 40,
                                    temperature = 0.7,
                                    n = 16)
  
  def _generate(self, prompts, mode = "generation"):
    if mode == "eval":
      outputs = self.model.generate(prompts,
                              self.eval_params
                              )
    else:
      outputs = self.model.generate(prompts,
                              self.sampling_params
                              )
    
    return outputs
  
  def generate_batch(self, batch):
    prompts = [self.generation_prompt.format(question = x["question"]) for x in batch]
    try:
      start_time = time.perf_counter()
      outputs = self._generate(prompts)
      end_time = time.perf_counter()
    except Exception as e:
      print(e)
      return None, None
    
    total_time = end_time - start_time
    return outputs, total_time
  
  def evaluate_batch(self, batch, inputs):
    output_confidences = []
    for input_idx, input in enumerate(inputs):
        prompts = [self.eval_prompt.format(question = batch[input_idx]["question"], solution = input.outputs[idx].text) for idx in range(len(input.outputs))]
        try:
            start_time = time.perf_counter()
            outputs = self._generate(prompts, "eval")
            end_time = time.perf_counter()
            confidences = [math.exp(x.outputs[0].logprobs[1][self.token_mapping["TRUE"]])/
                                    (math.exp(x.outputs[0].logprobs[1][self.token_mapping["TRUE"]])+ math.exp(x.outputs[0].logprobs[1][self.token_mapping["FALSE"]])) if 
                                    (self.token_mapping["TRUE"] in x.outputs[0].logprobs[1] and self.token_mapping["FALSE"] in x.outputs[0].logprobs[1])
                                    else 0 for x in outputs]
            #evaluations = [(x.outputs[0].text, math.exp(max(x.outputs[0].logprobs[1].values()))) for x in outputs]
            output_confidences.append(confidences)
        except Exception as e:
            print(e)
            return None, None

    total_time = end_time - start_time
    return output_confidences, total_time
    
    
def run_inference(data_path,
        model_name,
        mode = "eval-top1",
        datapoint_start_idx = 0,
        datapoint_end_idx = -1,
        access_to_gold_truth = True):
  # load in the data
  dataset = Dataset.from_dict(load_gsm8k_data(data_path))
  generator = Generator(
                        model_name,
                        mode)
  evaluator = Evaluator()
  rewards = []
  times = []
  tokens_per_sec = []
  batch = []
  predictions = []
  save_outputs = 500 # only used if mode is data_generation
  if datapoint_end_idx == -1:
    datapoint_end_idx = len(dataset)

  if mode.startswith("data_gen"):
    data_save_dir = "generated_data"
    if not os.path.exists(data_save_dir):
      os.mkdir(data_save_dir)
  for i in range(datapoint_start_idx, datapoint_end_idx):
    batch.append(dataset[i])
    if len(batch) == generator.batch_size or i == datapoint_end_idx -1:
      output_confidences = [[] for x in range(len(batch))]
      outputs, total_time = generator.generate_batch(batch)
      if mode in ["eval-conf_top1",
                  "eval-conf_classifier_maj_voting",
                  "data_gen-top_1_confidence_threshold",
                  "data_gen-conf_classifier_maj_voting"]:
        output_confidences, eval_time = generator.evaluate_batch(batch, outputs)

      if outputs is None:
        batch = []
        continue
      generated_tokens = 0
      for output in outputs:
        generated_tokens += sum([len(x.token_ids) for x in output.outputs])
      times.append(total_time)

      tokens_per_sec.append(generated_tokens/total_time)
      # print out gpu memory specs
      t = torch.cuda.get_device_properties(0).total_memory
      r = torch.cuda.memory_reserved(0)
      a = torch.cuda.memory_allocated(0)
      f= r-a  # free inside reserved
      info = nvmlDeviceGetMemoryInfo(h)
      print(f'total    : {info.total/ 1024 / 1024}')
      print(f'free     : {info.free/ 1024 / 1024}')
      print(f'used     : {info.used/ 1024 / 1024}')
      print(f"GPU utilisation {torch.cuda.memory_allocated(0)/ 1024 / 1024 / 1024 }")

      for idx in range(len(batch)):

        datapoint_solutions = [x.text for x in outputs[idx].outputs]
        chosen_answer_dict, reward = evaluator.evaluate_batch(datapoint_solutions,
                                              batch[idx]["answer"],
                                              mode,
                                              output_confidences[idx],
                                              access_to_gold_truth = access_to_gold_truth)
        if reward == -1:
           continue
        if mode.startswith("data_gen"):
            if chosen_answer_dict is not None and len(chosen_answer_dict[1]) > 0:
                prediction = {"correct_prediction": chosen_answer_dict[1],
                               "incorrect_prediction": chosen_answer_dict[0],
                            "question": batch[idx]["question"],
                            "answer": batch[idx]["answer"]}
                predictions.append(prediction)

            if len(predictions) >= save_outputs: # save every save_outputs datapoints
                with open(f"{data_save_dir}/correct_pred_{i}.json", "w", encoding = "utf-8") as final:
                    json.dump(predictions, final)
                predictions = []

        rewards.append(reward)
        print(reward)

      batch = []


  # print some summary statistics
  print(f"Average reward: {sum(rewards)/len(rewards)}")
  print(f"Number of datapoints used: {len(rewards)/(datapoint_end_idx - datapoint_start_idx)}")
  print(f"Average request time {sum(times)/len(times)}")
  print(f"Time per datapoint {sum(times)/(datapoint_end_idx - datapoint_start_idx)}")
  print(f"Tokens per second {sum(tokens_per_sec)/len(tokens_per_sec)}")