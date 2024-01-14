


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import wandb
import os
from torch.optim import AdamW
from datasets import Dataset
from data_utils import load_gsm8k_data, load_synthetic_data, DatasetProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import get_scheduler
from tqdm import tqdm
import torch

HF_TOKEN = ""
# login into wandb with wandb login
class TrainingArgs:
    def __init__(self,
                 per_device_train_batch_size = 4,
                 gradient_accumulation_steps = 10,
                 learning_rate = 2e-5,
                 weight_decay = 0.01,
                 num_train_epochs = 1,
                 eval_steps = 1000000,
                 lora_r = 128,
                 lora_alpha = 256,
                 lora_target_modules = [
                     "Wqkv",
                     "out_proj",
                 ],
                 dpo_beta = 0.1):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.eval_steps = eval_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.dpo_beta = dpo_beta

class Trainer:
    def __init__(self,
                 model,
                 training_args,
                 tokenizer,
                 train_dataloader):
        self.model = model
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.training_args = training_args
        self.num_training_steps = training_args.num_train_epochs * len(train_dataloader)
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin,
                                  mixed_precision="fp16"
                                  )
        self.initialise_optimizer()
        # todo apply peft to model and prepare acceleratro, wandb and model saving and uplodaing
        # also all "model.to.device" should be accceleratir.to_device
    
    def calc_loss(self,
                    inputs,
                    logits,
                    mode = "max_likelihood",
                    average_log_prob = False):
        # Shift so that tokens < n predict n
        shift_labels = inputs["input_ids"][..., 1:].contiguous().to(self.model.device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_token_type_ids = inputs["token_type_ids"][..., 1:].contiguous().to(self.model.device)
        if mode == "max_likelihood":
            # Calculate per-token loss
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # do not count loss for where the token_type_ids are not 1
            loss = loss * (flattened_token_type_ids == 1).float()
            flattened_token_type_ids = shift_token_type_ids.flatten()
            # Resize and average loss per sample
            loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
            # Calculate weighted average
            weighted_loss = loss_per_sample.mean()
            return weighted_loss
        elif mode == "dpo":
            logprobs = self.calculate_logprobs(shift_logits, shift_labels, shift_token_type_ids, average_log_prob)
            # split the logprobs such that every even is "chosen" and every odd is "not chosen"
            policy_chosen_logprobs = logprobs[::2]
            policy_not_chosen_logprobs = logprobs[1::2]
            reference_logits = inputs["reference_logits"].to(self.model.device)
            reference_logprobs = self.calculate_logprobs(reference_logits, shift_labels, shift_token_type_ids, average_log_prob)
            # split the logprobs such that every even is "chosen" and every odd is "not chosen"
            reference_chosen_logprobs = reference_logprobs[::2]
            reference_not_chosen_logprobs = reference_logprobs[1::2]
            policy_ratio = policy_chosen_logprobs - policy_not_chosen_logprobs
            reference_ratio = reference_chosen_logprobs - reference_not_chosen_logprobs
            policy_reference_difference = policy_ratio - reference_ratio
            loss = -F.logsigmoid(self.training_args.dpo_beta * policy_reference_difference).mean()
            return loss

    def calculate_logprobs(self, logits, labels, token_type_ids, average_log_prob):
        # get the likelihood of the target token
        per_token_likelihoods = torch.gather(logits.log_softmax(-1), dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
        masked_per_token_likelihoods = per_token_likelihoods * (token_type_ids == 1).float()
        if average_log_prob:
            return (masked_per_token_likelihoods).sum(-1) / token_type_ids.sum(-1)
        else:
            return (masked_per_token_likelihoods).sum(-1)
    
    def _get_grouped_params(self, model, no_decay=["bias", "LayerNorm.weight"]):
        params_with_wd, params_without_wd = [], []
        for n, p in model.named_parameters():
            if any(nd in n for nd in no_decay):
                params_without_wd.append(p)
            else:
                params_with_wd.append(p)
        return [
            {"params": params_with_wd, "weight_decay": self.training_args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]
    def initialise_optimizer(self):
        self.optimizer = AdamW(self._get_grouped_params(self.model),
                                lr=self.training_args.learning_rate,
                                weight_decay=self.training_args.weight_decay)



        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer = self.optimizer,
            num_warmup_steps=self.num_training_steps * 0.1,
            num_training_steps=self.num_training_steps,
        )
    def train(self):
        self.model.train()
        completed_steps = 0
        for epoch in range(self.training_args.num_train_epochs):
            for step, batch in tqdm(
                enumerate(self.train_dataloader, start=1), total=self.num_training_steps
            ):
                logits = self.model(batch["input_ids"].to(self.model.device)).logits
                loss = self.calc_loss(batch, logits, "dpo", True)
                if step % 10 == 0:
                    self.accelerator.print(
                        {
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "samples": step * self.training_args.per_device_train_batch_size,
                            "steps": completed_steps,
                            "loss/train": loss.item() * self.training_args.gradient_accumulation_steps,
                        }
                    )
                    #wandb.log({"lr": self.optimizer.param_groups[0]["lr"],
                    #           "loss/train": loss.item() * self.training_args.gradient_accumulation_steps})
                loss = loss / self.training_args.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if step % self.training_args.gradient_accumulation_steps == 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    completed_steps += 1
                #if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                #    eval_loss, perplexity, avg_reward, predictions = evaluate()
                #    accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity, "avg_reward": avg_reward})
                #    language_model.train()
                #    accelerator.wait_for_everyone()
                #    unwrapped_model = accelerator.unwrap_model(language_model)
                #    # make a new directory in the output dir with current step
                #    if intermediate_eval:
                #      step_output_dir = os.path.join(output_dir, f"step_{completed_steps}")
                #      os.mkdir(step_output_dir)
                #      # save the predictions
                #      with open(os.path.join(step_output_dir, "predictions.json"), "w") as f:
                #          json.dump(predictions, f)
                #    if accelerator.is_main_process and save_model:
                #        model.tokenizer.save_pretrained(step_output_dir)
                #        unwrapped_model.save_pretrained(step_output_dir, save_function=accelerator.save)
        #wandb.finish()
def main():
    
    model_name = "microsoft/phi-2"
    model_name = "EleutherAI/pythia-70m-v0"
    run_name = "wandb_run_name" # WANDB run name
    repo_name = "HF_repo_name" # HF repo name where model is saved
    data_mode = "original" # either use original or synthetic data loading
    data_path = r"" # where data is located


    if data_mode == "original":
        dataset = Dataset.from_dict(load_gsm8k_data(data_path))
    elif data_mode == "synthetic":
        dataset = Dataset.from_dict(load_synthetic_data(data_path))
    
    model = AutoModelForCausalLM.from_pretrained(
          model_name, device_map = "auto",trust_remote_code=True,
          token = HF_TOKEN,
          #torch_dtype="auto",
          #revision = "834565c23f9b28b96ccbeabe614dd906b6db551a"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
      model_name, device_map = "auto",
      #padding_side="left",
      add_eos_token=True,
      token = HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    dataset_processor = DatasetProcessor(tokenizer, r"prompts\finetuned_prompt.txt")
    dataset = dataset.shuffle(seed = 42)
    tokenized_dataset = dataset.map(dataset_processor.training_preprocessor_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")
    training_args = TrainingArgs()
    train_dataloader = DataLoader(tokenized_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    print(len(train_dataloader))
    trainer = Trainer(model,
                    training_args,
                    tokenizer,
                    train_dataloader)
    trainer.train()


if __name__ == "__main__":
    main()