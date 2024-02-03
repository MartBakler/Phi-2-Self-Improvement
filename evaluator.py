

class Evaluator():
    def __init__(self,
                pos_eval_confidence = 0.75,
                neg_eval_confidence = 0.6,
                majority_vote_confidence = 6):
        self.pos_eval_confidence = pos_eval_confidence
        self.neg_eval_confidence = neg_eval_confidence
        self.majority_vote_confidence = majority_vote_confidence


    def evaluate_batch(self, candidates, reference, mode, evaluations = None):
        if mode in ["eval-top16", "eval-top1"]:
          reward  = self.calculate_reward(candidates, reference)
          return {}, reward
        elif mode == "eval-majority_vote":
          chosen_answer_dict = self.get_majority_answer(candidates)
          reward = self.calculate_reward(chosen_answer_dict[1],
                             reference, 
                             )
          return {}, reward
        elif mode == "eval-conf_top1":
            ordered_candidates = sorted([(candidates[idx], evaluations[idx]) for idx in range(len(candidates))], key = lambda x: x[1], reverse = True)
            reward = self.calculate_reward([ordered_candidates[0][0]],
                             reference, 
                             )
            return {}, reward
    
        elif mode == "eval-conf_classifier_maj_voting":
            ordered_candidates = sorted([(candidates[idx], evaluations[idx]) for idx in range(len(candidates))], key = lambda x: x[1], reverse = True)
            filtered_candidates = [candidate[0] for candidate in ordered_candidates if candidate[1] > 0.5]
            if len(filtered_candidates) == 0:
                reward = self.calculate_reward([ordered_candidates[0][0]],
                             reference, 
                             )
                return {}, reward
            chosen_answer_dict = self.get_majority_answer(filtered_candidates)
            reward = self.calculate_reward(chosen_answer_dict[1],
                             reference, 
                             )
            return {}, reward
        
        elif mode == "data_gen-majority_vote":
          chosen_answer_dict = self.get_majority_answer(candidates)
          if len(chosen_answer_dict[1]) < self.majority_vote_confidence:
             return {}, -1
          reward = self.calculate_reward(chosen_answer_dict[1],
                             reference, 
                             )
          return chosen_answer_dict, reward
        
        elif mode == "data_gen-top_1_confidence_threshold":
          ordered_candidates = sorted([(candidates[idx], evaluations[idx]) for idx in range(len(candidates))], key = lambda x: x[1], reverse = True)
          if ordered_candidates[0][1] > self.pos_eval_confidence:
                chosen_answer_dict = {1: [ordered_candidates[0][0]], 0: [ordered_candidates[-1][0]]}
                reward = self.calculate_reward(chosen_answer_dict[1],
                             reference, 
                             )
                return chosen_answer_dict, reward
          return {}, -1
        
        elif mode == "data_gen-conf_classifier_maj_voting":
            ordered_candidates = sorted([(candidates[idx], evaluations[idx]) for idx in range(len(candidates))], key = lambda x: x[1], reverse = True)
            filtered_candidates = [candidate[0] for candidate in ordered_candidates if candidate[1] > 0.5]

            if len(filtered_candidates) == 0 or ordered_candidates[0][1] < self.pos_eval_confidence:
                  return {}, -1
            chosen_answer_dict = self.get_majority_answer(filtered_candidates)
            if len(chosen_answer_dict[1]) < self.majority_vote_confidence:
             return {}, -1
            reward = self.calculate_reward(chosen_answer_dict[1],
                               reference, 
                               )
            if len(chosen_answer_dict[0]) == 0: # if no candidates are left for the negative class
              # loop through the ordered candidates in reverse and add the first one that is not in the majority vote
              for candidate in ordered_candidates[::-1]:
                reward_to_majority = self.calculate_reward([candidate], chosen_answer_dict[1][0])
                if reward_to_majority == 0:
                  chosen_answer_dict[0].append(candidate[0])
                  break
            return chosen_answer_dict, reward


        #  elif strategy == "majority&confidence":
        #    return get_reward(filtered_candidates,
        #                       reference, 
        #                       strategy = strategy,
        #                       evaluations = evaluations,
        #                       access_to_gold_truth=access_to_gold_truth)

    def calculate_reward(self, candidates, reference):
        """
        Compute the reward for a candidate sentence given a reference sentence.
        """

        true_answer = reference.split("FINAL ANSWER:")[1].strip()
        for candidate in candidates:
            if "FINAL ANSWER:" not in candidate:
              continue
            else:
                answer = candidate.split("FINAL ANSWER:")[1]
                final_answer = answer.split("\n")[0].strip()
                if final_answer == true_answer:
                    return 1
        return 0
    

    def get_majority_answer(self, candidates):
        """
        This function is used to choose the generated data to save
        """
    
        chosen_answer_dict = {}
        answer_counts = {}
        for candidate in candidates:
          if "FINAL ANSWER:" not in candidate:
            continue
          else:
            answer = candidate.split("FINAL ANSWER:")[1]
            final_answer = answer.split("\n")[0].strip()
            if final_answer not in answer_counts:
              answer_counts[final_answer] = []
            answer_counts[final_answer].append(candidate)
        # # check that answer_counts isnt empty sequence
        if len(answer_counts) == 0:
          return {1: [], 0: []}
        most_common_answer = max(answer_counts, key=lambda x: len(answer_counts[x]))
        chosen_answer_dict[1] = answer_counts[most_common_answer]
        chosen_answer_dict[0] = [x for x in candidates if x not in chosen_answer_dict[1]]
        return chosen_answer_dict