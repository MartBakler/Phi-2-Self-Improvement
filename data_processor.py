

import os
import json
import re


def get_reward(candidate, reference):
    """
    Compute the reward for a candidate sentence given a reference sentence.
    """
    
    if "FINAL ANSWER:" not in candidate:
      return 0
    else:
      answer = candidate.split("FINAL ANSWER:")[1]
      final_answer = answer.split("\n")[0].strip()
      
      true_answer = reference.split("FINAL ANSWER:")[1].strip()
      if final_answer == true_answer:
          return 1
      else:
          return 0



def load_data_new_format(data_path):
    # get all files in data_path
    files = os.listdir(data_path)
    dataset = []
    rewards = []
    remove_calc = True
    questions = []
    for file in files:
      dataset_file = os.path.join(data_path, file)
      # read the json file
      with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
      for datapoint in data:
        if datapoint["question"] in questions:
          continue
        questions.append(datapoint["question"])
        correct_predicions = []
        incorrect_predictions = []
        for prediction in datapoint["correct_prediction"]:
          if "FINAL ANSWER:" not in prediction:
              continue
          answer = prediction.split("FINAL ANSWER:")[1].strip()
          final_answer = answer.split("\n")[0].strip()
          prefix = prediction.split("FINAL ANSWER:")[0]
          prefix = prefix.strip().rstrip("\n")
          final_prediction = prefix + "\nFINAL ANSWER: " + final_answer
          final_prediction = final_prediction.replace("# Solution\n", "").replace("# Solution:\n", "")
          if remove_calc:
             #remove everything between < and > 
              final_prediction = re.sub("<<.*?>>", "", final_prediction)
          correct_predicions.append(final_prediction)
          reward = get_reward(final_prediction, datapoint["answer"])
          rewards.append(reward)
        for prediction in datapoint["incorrect_prediction"]:
          if "FINAL ANSWER:" not in prediction:
              continue
          answer = prediction.split("FINAL ANSWER:")[1].strip()
          final_answer = answer.split("\n")[0].strip()
          prefix = prediction.split("FINAL ANSWER:")[0]
          prefix = prefix.strip().rstrip("\n")
          final_prediction = prefix + "\nFINAL ANSWER: " + final_answer
          final_prediction = final_prediction.replace("# Solution\n", "").replace("# Solution:\n", "")
          reward = get_reward(final_prediction.replace("$", ""), datapoint["answer"]) # calculate reward without $ signs to check if the answer is correct
          if reward == 1:
            continue
          if remove_calc:
             #remove everything between < and > 
              final_prediction = re.sub("<<.*?>>", "", final_prediction)
          incorrect_predictions.append(final_prediction)

        dataset.append({"question": datapoint["question"],
                         "original_answer": datapoint["answer"],
                          "correct_predictions": correct_predicions,
                          "incorrect_predictions": incorrect_predictions})
        
      #with open(dataset_file, "r") as f:
      #  for line in f:
      #      line = json.loads(line)
      #      data["question"].append(line["question"])
      #      data["answer"].append(line["answer"].replace("####", "FINAL ANSWER:"))
    assert(list(set(rewards)) == [1])
    return dataset


data_path = r""
dataset = load_data_new_format(data_path)
# save the dataset into the data path as a jsonl
with open(os.path.join(data_path, "combined_dataset.jsonl"), "w", encoding="utf-8") as f:
  for datapoint in dataset:
    json.dump(datapoint, f)
    f.write("\n")
