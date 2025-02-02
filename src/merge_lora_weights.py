import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, PeftModel

def merge_lora(args):

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_base,
        torch_dtype='auto',
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_base)
    peft_model_name = args.lora_path
    merged_model= PeftModel.from_pretrained(model,peft_model_name)
    merged_model= merged_model.merge_and_unload()
    model.save_pretrained(args.save_model_path)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)