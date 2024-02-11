# Phi-2-Self-Training

## Introduction 

This is repo for experimenting with the self-training capabilities of the Microsoft Phi-2 language model. The different experiments are brought out in their respective sections and all are ran on the GSM8K dataset using the original Phi-2 model. There have been multiple papers on different self-training methodologies (brought out below) but none of them on this small scale (2.7B model). Training was done with Pytorch, Peft (for applying lora weights) and HF Accelerate,  and inference was done with VLLM and some custom sampling code for GMS8K

Note: The original numbers Phi-2 reported on GMS8K from the blogpost are using code generation. I do not use code generation or code evaluation in any of the experiments, all calculations are needed to be done by the LLM itself


## Experiment 1 - Training Phi 2 on its own answer generations

This experiment follows the paper "Beyond Human Data: Scaling Self-Training for
Problem-Solving with Language Models" ([Arxiv](https://arxiv.org/pdf/2312.06585.pdf)) By Google & Google Deepmind but with a much smaller langauge model.

First using the original Phi-2 in a 5-shot manner, a dataset of 1000 synthetic mathematical solutions are generated using the GSM8K training questions. The dataset is created by letting the original model synthesise answers to the questions and only the correctness of the final answer is checked, if it is correct, it is added to the fataset. This is like rejection sampling with a reward model, which will check if the final answer of the completion is correct. Then the Phi-2 model is trained on the 1000 synthetic datapoints (Dataset 1) and the resulting model is used to generate more training data (6000 datapoints, Dataset 2). In the second iteration incorrect datapoints are also collected for DPO finetuning. 

The results are seen below for the following sampling strategies -- best-of-1, best-of-16, majority vote (16 solution samplings per question, generations are grouped by their final answer) 



# Evaluation
| Model              |  Sampling strategy  | Accuracy (test set)|
| ------------------ | ------------------  |--------------------|
|Original Phi 2 model| 5-shot, best of 16  | 66%                |
|Original Phi 2 model| 5-shot, best of 1   | 37%                |
|SFT Dataset 1       | 0-shot, best of 16  | 87%                |
|SFT Dataset 1       | 0-shot, best of 1   | 58%                |
|SFT Dataset 1       | majority vote (16)  | 69%                |
|SFT Dataset 2       | 0-shot, best of 1   | 64%                |
|SFT Dataset 2       | majority vote (16)  | 72%                |
|DPO (Dataset 2)     | 0-shot, best of 1   | TBD%                 |
|DPO (Dataset 2)     | majority vote (16)  | TBD%                 |


**Interesting results**: As can be seen, only training on the synthetic solutions does increase the performance on test set significantly and steadily. Thus the results shown in the above paper do generalise to smaller models as well.

## Experiment 2 - Training Phi 2 on its own answer generations and evaluations 

This experiment combines the output answer generation and evaluation of correct and incorrect answers. Lately a couple of papers have been testing the same premise and have been showing good results ([Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020.pdf), [Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs](https://aclanthology.org/2023.findings-emnlp.345.pdf)) 

Similarly to the first experiment, using the original Phi-2 in a 5-shot manner, a dataset of 1000 synthetic mathematical solutions are generated using the GSM8K training questions. For each question, both correct and incorrect generations are saved. Then the language model is trained in a SFT way on 2 tasks - Generation of correct answers to a question and labelling answer traces either correct or incorrect. This essentially results in a model being a generator and also a reward model. 

The Phi-2 model is first trained on the 1000 synthetic datapoints (Dataset 1) and the resulting model is used to generate more training data (6000 datapoints, Dataset 2) in the same manner. Moreover a third and a fourth dataset is collected consisting of only high confidence solutions (Dataset 3) and high and low confidence solutions (Dataset 4), the model trained on Dataset 1 both generated and evaluated the solutions without checking the final answer (fully synthetic collection without the need for ground truths). The models trained on Dataset 3 are trained on high confidence solutions following the classical GSM8K SFT target and models trained on Dataset 4 are trained both on the classical task and also on the "evaluation/reward modelling" task, where Correct and Incorrect labels are gotten from the high and low confidence solutions respectively. 

The results are seen below for the following sampling strategies -- best-of-1, best-of-16, majority vote (16 samplings per question, generations are grouped by their final answer), highest confidence best of 1 (16 solution generation per question, getting the same model to evaluate the answers and choosing the one with the highest "Correct" logit normalised score), majority vote + evaluation filter (16 solution generation per question, getting the same model to evaluate each of the answers, removing the solutions that are evaluated as "Incorrect" and then doing majority voting on remainder of samples)



# Evaluation
| Model            |          Generation strategy           | Accuracy (test set)|
| -----------------| ---------------------------------------|--------------------|
|SFT Dataset 1     | best of 16                             | 87%                |
|SFT Dataset 1     | best of 1                              | 57%                |
|SFT Dataset 1     | majority vote (16)                     | 69%                |
|SFT Dataset 1     | majority vote (16) + evaluation filter | 71%                |
|SFT Dataset 1     | highest confidence best of 1 (16)      | 59%                |
|SFT Dataset 2     | 0-shot, best of 1                      | 62%                |
|SFT Dataset 2     | majority vote (16)                     | 72%                |
|SFT Dataset 2     | majority vote (16) + evaluation filter | 75%                |
|SFT Dataset 2     | highest confidence best of 1 (16)      | 69%                |
|SFT Dataset 3     | 0-shot, best of 1                      | 63%                |
|SFT Dataset 3     | majority vote (16)                     | 73%                |
|SFT Dataset 4     | 0-shot, best of 1                      | 63%                |
|SFT Dataset 4     | majority vote (16)                     | 72%                |
|SFT Dataset 4     | majority vote (16) + evaluation filter | 74%                |
|SFT Dataset 4     | highest confidence best of 1 (16)      | 67%                |
|DPO (Dataset 4)   | 0-shot, best of 1                      | TBD%               |
|DPO (Dataset 4)   | majority vote (16)                     | TBD%               |


**Interesting results**: Multiple interesting results can be seen here

* Training on solution generation and evaluation tasks at the same time minimally reduces model performance on the solution generation task compared to training only on that one task

* Using the evaluation confidences does increase performance of top-1 answer evaluation (64% --> 69%) and majority voting evaluation (72%--> 75%) 

* Using only the high confidence outputs as a reward signal (fully synthetic data) after finetuning on the multi-task target is essentially the same as training on the "correct" solutions (majority vote 72 vs 73, top-1 64 vs 63). This shows this method could be used to scale up the datasets massively and increase performance by training on only synthetic data. It also shows the method of self-evaluation does work on a smaller language model only having 2.7B parameters
