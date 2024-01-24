


def get_reward(candidates, reference, strategy = "evaluate_all", evaluations = None, access_to_gold_truth = True):
    """
    Compute the reward for a candidate sentence given a reference sentence.
    """
    chosen_answer_dict = {0:[], 1:[]}
    answer_counts = {}
    answer_mean_evaluations = {}
    for candidate in candidates:
        if "FINAL ANSWER:" not in candidate:
          continue
        else:
          answer = candidate.split("FINAL ANSWER:")[1]
          final_answer = answer.split("\n")[0].strip()
          if final_answer not in answer_counts:
            answer_counts[final_answer] = []
            answer_mean_evaluations[final_answer] = 1
          answer_counts[final_answer].append(candidate)
          if evaluations is not None:
            confidence = evaluations[candidates.index(candidate)][1] if evaluations[candidates.index(candidate)][0] == "True" else 1 - evaluations[candidates.index(candidate)][1]
            answer_mean_evaluations[final_answer] += confidence
        
        true_answer = reference.split("FINAL ANSWER:")[1].strip()
        if access_to_gold_truth:
          if final_answer == true_answer:
              chosen_answer_dict[1].append(candidate)
          else:
              chosen_answer_dict[0].append(candidate)
    if strategy == "evaluate_all":
      if not access_to_gold_truth:
         raise ValueError("Cannot use evaluate_all without access to gold truth")
      reward = 1 if len(chosen_answer_dict[1]) > 0 else 0
    elif strategy == "majority_vote":
      # get the most common answer
      most_common_answer = max(answer_counts, key=lambda x: len(answer_counts[x]))
      reward = 1 if most_common_answer == true_answer else 0 
      if not access_to_gold_truth:
        chosen_answer_dict[1] = answer_counts[most_common_answer]
        chosen_answer_dict[0] = [x for x in candidates if x not in chosen_answer_dict[1]]
    elif strategy == "majority&confidence":
      answer_final_scores = {}
      for answer, answer_candidates in answer_counts.items():
        answer_final_scores[answer] = (answer_mean_evaluations[answer]/len(answer_candidates)) * len(answer_candidates)/len(candidates)
        winning_answer = max(answer_final_scores, key=lambda x: answer_final_scores[x])
        reward = 1 if winning_answer == true_answer else 0
        if not access_to_gold_truth:
          chosen_answer_dict[1] = answer_counts[winning_answer]
          chosen_answer_dict[0] = [x for x in candidates if x not in chosen_answer_dict[1]]
    return chosen_answer_dict, reward


def evaluate_batch(candidates, reference, mode, eval_confidence, evaluations = None, access_to_gold_truth = True):
    if mode in ["eval@16", "eval@1"]:
      return get_reward(candidates,
                         reference, 
                         strategy="evaluate_all",
                         evaluations = evaluations,
                         access_to_gold_truth = access_to_gold_truth)
    elif mode == "evaluation_majority_vote":
      return get_reward(candidates,
                         reference, 
                         strategy = "majority_vote",
                         evaluations = evaluations,
                         access_to_gold_truth = access_to_gold_truth)
    elif mode.startswith("evaluation_generation_with_eval"):
      strategy = mode.split("_")[-1]
      if strategy == "filter":
        filtered_candidates = [candidates[idx] for idx in range(len(candidates)) if "True" in evaluations[idx][0] and evaluations[idx][1] > eval_confidence]
        if len(filtered_candidates) == 0:
           return None, 0
        return get_reward(filtered_candidates, 
                          reference, 
                          strategy = "majority_vote",
                          evaluations = evaluations,
                          access_to_gold_truth = access_to_gold_truth)
      elif strategy == "order":
        #order candidates by confidence, results in a list of tuples (candidate, evalutation, confidence)
        ordered_candidates = sorted([(candidates[idx], evaluations[idx][0], evaluations[idx][1]) for idx in range(len(candidates))], key = lambda x: x[2], reverse = True)
        # take the first candidate that has True
        filtered_candidates = [x[0] for x in ordered_candidates if "True" in x[1]]
        if len(filtered_candidates) == 0:
           return None, 0
        filtered_candidates = [filtered_candidates[0]]
        return get_reward(filtered_candidates,
                           reference, 
                           strategy = "evaluate_all",
                           evaluations = evaluations,
                           access_to_gold_truth=access_to_gold_truth)
      elif strategy == "majority&confidence":
        return get_reward(filtered_candidates,
                           reference, 
                           strategy = strategy,
                           evaluations = evaluations,
                           access_to_gold_truth=access_to_gold_truth)
