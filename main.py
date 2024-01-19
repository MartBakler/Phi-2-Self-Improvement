import argparse
from model_inference import run_inference
from train import train_model
def get_args():

    parser = argparse.ArgumentParser()
    #args used for both inference and training
    parser.add_argument("--model_name", type=str, default="/path/to/model")
    parser.add_argument("--dataset_path", type=str, default="/path/to/data")
    parser.add_argument("--prompt_path", type=str, default="/path/to/prompt_for_inference")
    #args used for inference
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--eval_datapoint_start", type=int, default=0)
    parser.add_argument("--eval_datapoint_finish", type=int, default=-1)

    # args used for training
    parser.add_argument("--upload_model", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default="default_project")
    parser.add_argument("--wandb_run_name", type=str, default="default_training_run")
    parser.add_argument("--hf_repo_name", type=str, default="default_finetuned_model")
    parser.add_argument("--dataset_type", type=str, default="original")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=256)
    parser.add_argument("--lora_target_modules", type=list, default=[
                     "Wqkv",
                     "out_proj",
                        ])
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--lr_schedule_name", type=str, default="cosine")
    parser.add_argument("--warmup_proportion", type=float, default=0.2)
    parser.add_argument("--mixed_prec_training", type=str, default="fp16")


    
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    args, unknown = get_args()
    print(args)
    if args.mode in ["eval", "data_generation"]:
        run_inference(args.dataset_path,
                      args.model_name,
                      args.prompt_path,
                      args.mode,
                      args.eval_datapoint_start,
                      args.eval_datapoint_finish)
    elif args.mode == "training":
        train_model(args) # a couple of TODOs left in train_model
    else:
        raise ValueError("Mode must be one of: eval, data_generation, training")
    


if __name__ == "__main__":
    main()