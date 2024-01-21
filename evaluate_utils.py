


def get_reward(candidates, reference, strategy = "all"):
    """
    Compute the reward for a candidate sentence given a reference sentence.
    """
    reward_dict = {0:[], 1:[]}
    answer_counts = {}
    for candidate in candidates:
        if "FINAL ANSWER:" not in candidate:
          reward_dict[0].append(candidate)
          continue
        else:
          answer = candidate.split("FINAL ANSWER:")[1]
          final_answer = answer.split("\n")[0].strip()
          if final_answer not in answer_counts:
            answer_counts[final_answer] = 0
          answer_counts[final_answer] += 1
          
        
        true_answer = reference.split("FINAL ANSWER:")[1].strip()
        if final_answer == true_answer:
            reward_dict[1].append(candidate)
        else:
            reward_dict[0].append(candidate)
    if strategy == "all":
      reward = 1 if len(reward_dict[1]) > 0 else 0
    elif strategy == "majority_vote":
      # get the most common answer
      most_common_answer = max(answer_counts, key=answer_counts.get)
      reward = 1 if most_common_answer == true_answer else 0 
    return reward_dict, reward


def evaluate_batch(candidates, reference, mode):
   if mode == "evaluation_generation_only":
      return get_reward(candidates, reference)
   elif mode == "evaluation_majority_vote":
      return get_reward(candidates, reference, strategy = "majority_vote")
      