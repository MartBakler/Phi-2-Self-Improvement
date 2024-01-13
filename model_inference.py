from vllm import LLM, SamplingParams
from data_utils import load_gsm8k_data
from reward_utils import get_reward
import json
import os
from datasets import Dataset
import torch
import time
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)


class Generator:
  def __init__(self,
              prompt_path : str,
              model_name : str,
              mode : str = "eval"):
      self.prompt_path = prompt_path
      self.prompt = self._load_prompt()
      self.model = LLM(model=model_name,
          trust_remote_code=True
          )
      self.mode = mode
      self.batch_size = 8 # specify generation batch size
      if mode == "eval":
         self.sampling_params = SamplingParams(max_tokens=384,
                                    top_k = 40,
                                    temperature = 0.3,
                                    n = 1,
                                    best_of = 5)
         
      elif mode == "data_generation":
         self.sampling_params = SamplingParams(max_tokens=384,
                                    top_k = 40,
                                    temperature = 0.7,
                                    n = 16)

  def _load_prompt(self):
      with open(self.prompt_path, "r", encoding = "utf-8") as f:
          prompt = f.read()
      return prompt
  
  def _generate(self, prompts):
      outputs = self.model.generate(prompts,
                              self.sampling_params
                              )
      return outputs
  
  def generate_batch(self, batch):
    prompts = [self.prompt.format(question = x["question"]) for x in batch]
    try:
      start_time = time.perf_counter()
      outputs = self._generate(prompts)
      end_time = time.perf_counter()
    except:
      return None, None
    
    total_time = end_time - start_time
    return outputs, total_time
    
    
def main(data_path,
        model_name,
        prompt_path, 
        mode = "eval",
        datapoint_start_idx = 0,
        datapoint_end_idx = -1):
  # load in the data
  dataset = Dataset.from_dict(load_gsm8k_data(data_path))
  generator = Generator(prompt_path,
                        model_name,
                        mode)
  rewards = []
  times = []
  tokens_per_sec = []
  batch = []
  correct_predictions = []
  save_outputs = 500 # only used if mode is data_generation
  if datapoint_end_idx == -1:
    datapoint_end_idx = len(dataset)

  if mode == "data_generation":
    data_save_dir = "generated_data"
    if not os.path.exists(data_save_dir):
      os.mkdir(data_save_dir)
  for i in range(datapoint_start_idx, datapoint_end_idx):
    batch.append(dataset[i])
    if len(batch) == generator.batch_size or i == datapoint_end_idx -1:
      outputs, total_time = generator.generate_batch(batch)
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
        reward_dict = get_reward(datapoint_solutions, batch[idx]["answer"])

        if mode == "data_generation":
          for rew_idx, reward in enumerate(list(reward_dict.values())):
            if reward == 1:
              correct_pred = {"prediction": list(reward_dict.keys())[rew_idx],
                              "question": batch[idx]["question"],
                              "answer": batch[idx]["answer"]}
              correct_predictions.append(correct_pred)

              if len(correct_predictions) >= save_outputs: # save every save_outputs datapoints
                  with open(f"{data_save_dir}/correct_pred_{i}.json", "w", encoding = "utf-8") as final:
                      json.dump(correct_predictions, final)
                  correct_predictions = []

        reward_values = list(reward_dict.values())
        reward = max(reward_values)
        rewards.append(reward)
        print(reward)

      batch = []


  # print some summary statistics
  print(sum(rewards)/len(rewards))
  print(f"Average request time {sum(times)/len(times)}")
  print(f"Time per datapoint {sum(times)/(datapoint_end_idx - datapoint_start_idx)}")
  print(f"Tokens per second {sum(tokens_per_sec)/len(tokens_per_sec)}")


if __name__ == "__main__":
    data_path = "data_where_you_saved_the_data.jsonl"
    model_name = "some_model_name"
    prompt_path = "prompts/finetuned_prompt.txt"
    main(data_path,
        model_name,
        prompt_path,
        mode = "eval",
        datapoint_start_idx = 0,
        datapoint_end_idx = 100,
        )
