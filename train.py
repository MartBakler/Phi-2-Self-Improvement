


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import wandb
import os
from torch.optim import AdamW
from datasets import Dataset
from data_utils import load_gsm8k_data, load_synthetic_data, DatasetProcessor, load_HF_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import get_scheduler
from tqdm import tqdm
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

#TODO
# wandb,
# model saving and uplodaing in fp16
# change in data format needs to be compatible with original data training


HF_TOKEN = ""

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# login into wandb with wandb login


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
                                  mixed_precision=self.training_args.mixed_prec_training
                                  )
        self.initialise_optimizer()
        self.prepare_peft_model()
        self.initialise_wandb
        self.model, self.optimizer, self.train_dataloader, = self.accelerator.prepare(
        self.model, self.optimizer, self.train_dataloader
            )
        


    def initialise_wandb(self):
        wandb.init(
                # set the wandb project where this run will be logged
                project=self.training_args.wandb_project_name,
                name = self.training_args.wandb_run_name,
                # track hyperparameters and run metadata
                config={
                "learning_rate": self.training_args.learning_rate,
                "weight_decay": self.training_args.weight_decay,
                "architecture": "Phi-2",
                "dataset": self.training_args.dataset_path,
                "epochs": self.training_args.num_train_epochs,
                "lora_r": self.training_args.lora_r,
                "lora_alpha": self.training_args.lora_alpha,
                "repo_name": self.training_args.hf_repo_name,
                }
            )
        
    def prepare_peft_model(self):
        self.model = prepare_model_for_kbit_training(self.model)
        config = LoraConfig(
            r=self.training_args.lora_r,
            lora_alpha=self.training_args.lora_alpha,
            target_modules=self.training_args.lora_target_modules,
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, config)
        print_trainable_parameters(self.model)

    def calc_loss(self,
                    inputs,
                    logits,
                    average_log_prob = False):
        # Shift so that tokens < n predict n
        if "sft" in self.training_args.training_mode:
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_token_type_ids = inputs["token_type_ids"][..., 1:]
            flattened_token_type_ids = shift_token_type_ids.flatten()
            # Calculate per-token loss
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # do not count loss for where the token_type_ids are not 1
            loss = loss * (flattened_token_type_ids == 1).float()
            # Resize and average loss per sample
            loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
            # Calculate weighted average
            weighted_loss = loss_per_sample.mean()
            return weighted_loss
        elif "dpo" in self.training_args.training_mode:
            # flatten the inputs

            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_token_type_ids = inputs["token_type_ids"][..., 1:]
            logprobs = self.calculate_logprobs(shift_logits, shift_labels, shift_token_type_ids, average_log_prob)
            # split the logprobs such that every the first half is "chosen" and the second half is "not chosen"
            policy_chosen_logprobs = logprobs[:logprobs.shape[0]//2]
            policy_not_chosen_logprobs = logprobs[logprobs.shape[0]//2:]
            reference_logits = inputs["reference_logits"]
            reference_logprobs = self.calculate_logprobs(reference_logits, shift_labels, shift_token_type_ids, average_log_prob)

            reference_chosen_logprobs = reference_logprobs[:reference_logprobs.shape[0]//2]
            reference_not_chosen_logprobs = reference_logprobs[reference_logprobs.shape[0]//2:]

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
            name=self.training_args.lr_schedule_name,
            optimizer = self.optimizer,
            num_warmup_steps=self.num_training_steps * self.training_args.warmup_proportion,
            num_training_steps=self.num_training_steps,
        )
    def train(self):
        self.model.train()
        completed_steps = 0
        if "dpo" in self.training_args.training_mode:
            ref_model = self.model.get_base_model()
        for epoch in range(self.training_args.num_train_epochs):
            for step, batch in tqdm(
                enumerate(self.train_dataloader, start=1), total=len(self.train_dataloader)
            ):
                if "dpo" in self.training_args.training_mode:
                    batch["input_ids"] = batch["input_ids"].flatten(0,1)
                    batch["token_type_ids"] = batch["token_type_ids"].flatten(0,1)
                    with torch.no_grad():
                        reference_logits = ref_model(batch["input_ids"]).logits
                        batch["reference_logits"] = reference_logits
                logits = self.model(batch["input_ids"]).logits
                loss = self.calc_loss(batch, logits, True)
                if step % 10 == 0:
                    self.accelerator.print(
                        {
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "samples": step * self.training_args.per_device_train_batch_size,
                            "steps": completed_steps,
                            "loss/train": loss.item() * self.training_args.gradient_accumulation_steps,
                            "epoch": epoch,
                        }
                    )
                    wandb.log({"lr": self.optimizer.param_groups[0]["lr"],
                               "loss/train": loss.item() * self.training_args.gradient_accumulation_steps})
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
        wandb.finish()
        if self.training_args.upload_model:
            self.upload_model()

    
    def upload_model(self):
        # save the model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.push_to_hub(self.training_args.hf_repo_name, token = HF_TOKEN, private = True)
        self.tokenizer.push_to_hub(self.training_args.hf_repo_name, token = HF_TOKEN)


def train_model(args):
    
    model_name = args.model_name
    data_type = args.dataset_type # either use original or synthetic data loading
    data_path = args.dataset_path # where data is located


    if data_type == "original":
        dataset = Dataset.from_dict(load_gsm8k_data(data_path))
    elif data_type in ["synthetic", "dpo"]:
        dataset = Dataset.from_dict(load_synthetic_data(data_path))
    #elif data_type == "hf":
    #    dataset = load_HF_data(data_path, HF_TOKEN)
    
    model = AutoModelForCausalLM.from_pretrained(
          model_name, device_map = "auto",trust_remote_code=True,
          #token = HF_TOKEN,
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
    dataset_processor = DatasetProcessor(tokenizer,
                                          r"prompts\sft_generation.txt",
                                          r"prompts\sft_eval.txt",
                                          args.training_mode)
    dataset = dataset.shuffle(seed = 42)
    if args.training_mode == "dpo":
         tokenized_dataset = dataset.map(dataset_processor.dpo_preprocessor_function, batched=True, remove_columns=dataset.column_names["train"])
    else:
        tokenized_dataset = dataset.map(dataset_processor.training_preprocessor_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format("torch")
    train_dataloader = DataLoader(tokenized_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    print(len(train_dataloader))
    trainer = Trainer(model,
                    args,
                    tokenizer,
                    train_dataloader)
    trainer.train()


if __name__ == "__main__":
    train_model()