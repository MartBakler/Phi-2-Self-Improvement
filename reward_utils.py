


def get_reward(candidates, reference):
    """
    Compute the reward for a candidate sentence given a reference sentence.
    """
    reward_dict = {}
    for candidate in candidates:
        if "FINAL ANSWER:" not in candidate:
          reward_dict[candidate] = 0
          continue
        else:
          answer = candidate.split("FINAL ANSWER:")[1]
          final_answer = answer.split("\n")[0].strip()
        
        true_answer = reference.split("FINAL ANSWER:")[1].strip()
        if final_answer == true_answer:
            reward_dict[candidate] = 1
            break
        else:
            reward_dict[candidate] = 0
    return reward_dict
