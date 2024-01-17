


def get_reward(candidates, reference):
    """
    Compute the reward for a candidate sentence given a reference sentence.
    """
    reward_dict = {0:[], 1:[]}
    for candidate in candidates:
        if "FINAL ANSWER:" not in candidate:
          reward_dict[0].append(candidate)
          continue
        else:
          answer = candidate.split("FINAL ANSWER:")[1]
          final_answer = answer.split("\n")[0].strip()
        
        true_answer = reference.split("FINAL ANSWER:")[1].strip()
        if final_answer == true_answer:
            reward_dict[1].append(candidate)
        else:
            reward_dict[0].append(candidate)
    return reward_dict
