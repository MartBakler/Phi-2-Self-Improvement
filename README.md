# Phi-2-Self-Training

## Introduction 

This is repo for application of "Beyond Human Data: Scaling Self-Training for
Problem-Solving with Language Models" ([Arxiv](https://arxiv.org/pdf/2312.06585.pdf)) By Google & Google Deepmind on GSM8K dataset using Phi 2 ([HF](https://huggingface.co/microsoft/phi-2)).

First using Phi-2 in a 5-shot manner, a dataset of synthetic examples is generated, where the original answers are only used to check if the final answer of synthetic data is correct. Then the Phi-2 model is trained on those synthetic datapoints and the resulting model is used to generate more training data. THis process is iterated 2 times and the results are seen below. 

NB: The original numbers Phi-2 reported from the paper are using code generation. I do not use code generation or code evaluation, all calculations are needed to be done by the LLM itself


## Evaluation
| Model              | Generation params  | Accuracy (test set)|
| ------------------ | ------------------ |--------------------|
|Original Phi 2 model| 5-shot, best of 16 | 47%                |
|Original Phi 2 model| 5-shot, best of 1  | 20%                |
|Finetune iteration 1| 0-shot, best of 16 | 87%                |
|Finetune iteration 1| 0-shot, best of 1  | 55%                |
|Finetune iteration 2| 0-shot, best of 1 | 63%                |


