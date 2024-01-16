import argparse
from model_inference import run_inference

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--dataset_path", type=str, default="/path/to/data")
    parser.add_argument("--prompt_path", type=str, default="/path/to/prompt_for_inference")
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--eval_datapoint_start", type=int, default=0)
    parser.add_argument("--eval_datapoint_finish", type=int, default=-1)


    
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    args, unknown = get_args()
    print(args)
    if args.mode in ["eval", "data_generation"]:
        run_inference(args.dataset_path,
                      args.model_path,
                      args.prompt_path,
                      args.mode,
                      args.eval_datapoint_start,
                      args.eval_datapoint_finish)
    elif args.mode == "training":
        pass # todo add training code
    else:
        raise ValueError("Mode must be one of: eval, data_generation, training")
    


if __name__ == "__main__":
    main()