import argparse
def get_args():

    parser = argparse.ArgumentParser()
    #args used for both inference and training
    parser.add_argument("--model_name", type=str, default="/path/to/model")
    parser.add_argument("--dataset_path", type=str, default="/path/to/data")
    #args used for inference
    parser.add_argument("--mode", type=str, default="training")
    parser.add_argument("--eval_datapoint_start", type=int, default=0)
    parser.add_argument("--eval_datapoint_finish", type=int, default=-1)

    # args used for training
    parser.add_argument("--upload_model", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default="default_project")
    parser.add_argument("--wandb_run_name", type=str, default="default_training_run")
    parser.add_argument("--hf_repo_name", type=str, default="default_finetuned_model")
    parser.add_argument("--dataset_type", type=str, default="synthetic")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=256)
    parser.add_argument("--lora_target_modules", type=list, default=[
            "q_proj",
             "k_proj",
             "v_proj"
             "dense",
        ])
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--lr_schedule_name", type=str, default="cosine")
    parser.add_argument("--warmup_proportion", type=float, default=0.2)
    parser.add_argument("--mixed_prec_training", type=str, default="fp16")
    parser.add_argument("--training_mode", type=str, default="sft_gen")


    
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    args, unknown = get_args()
    if args.mode in [   "eval-top1", # evaluate only 1 prediction
                        "eval-top16", # evaluate all 16 predictions
                        "eval-majority_vote", # get the majority vote for answer from the 16 predictions
                        "eval-conf_top1", # evaluate with the top 1 prediction with the highest confidence out of the 16 predictions
                        "eval-conf_classifier_maj_voting", # evaluate only with True out of 16 and then do majority vote
                        "data_gen-majority_vote", # generate data with majority vote and majority threshold
                        "data_gen-top_1_confidence_threshold", # generate data with confidence threshold
                        "data_gen-conf_classifier_maj_voting"]: # generate data with confidence threshold and majority vote
        from model_inference import run_inference
        run_inference(args.dataset_path,
                      args.model_name,
                      args.mode,
                      args.eval_datapoint_start,
                      args.eval_datapoint_finish,
                    )
    elif args.mode == "training":
        from train import train_model
        train_model(args) # a couple of TODOs left in train_model
    else:
        raise ValueError("Mode must be one of: eval-top1, eval-top16, eval-majority_vote, eval-conf_top1 eval-conf_classifier_maj_voting, data_gen-majority_vote, data_gen-top_1_confidence_threshold, data_gen-conf_classifier_maj_voting, training")
    


if __name__ == "__main__":
    main()