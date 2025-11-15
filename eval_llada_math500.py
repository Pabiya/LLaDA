# eval_math500_llada.py
import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from generate import generate  # MIRAGE가 들어있는 LLaDA generate
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


def build_model_and_tokenizer(model_path: str, device: torch.device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, cache_dir="./cache"
        )
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from {model_path}: {e}")
        print("[WARN] Falling back to GSAI-ML/LLaDA-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, cache_dir="./cache"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir="./cache",
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def parse_solution(solution: str):
    gold_parsed = parse(
        solution,
        extraction_config=extract_config,
    )
    if len(gold_parsed) == 0:
        gold_parsed = parse(
            "$" + solution + "$",
            extraction_config=extract_config,
        )
    return gold_parsed


def build_chat_prompts(tokenizer, problems, system_prompt_type="normal"):
    """MATH-500 문제들을 LLaDA 채팅 템플릿으로 감싸기"""
    prompts = []
    if system_prompt_type == "normal":
        system_prompt = (
            "Solve this math problem step by step, "
            "and put the final answer in \\boxed{}."
        )
    else:
        # 필요하면 다른 프롬프트 타입도 추가해서 쓸 수 있음
        system_prompt = (
            "Solve this math problem and return the final answer in \\boxed{}."
        )

    for prob in problems:
        user_content = prob + "\n" + system_prompt
        msgs = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)
    return prompts


def collate_math(batch, tokenizer, system_prompt_type="normal"):
    problems = [b["problem"] for b in batch]
    solutions = [b["solution"] for b in batch]
    levels = [b.get("level", None) for b in batch]
    types = [b.get("type", None) for b in batch]

    prompts = build_chat_prompts(tokenizer, problems, system_prompt_type=system_prompt_type)
    enc = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "problems": problems,
        "solutions": solutions,
        "levels": levels,
        "types": types,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="HuggingFaceH4/MATH-500",
        choices=[
            "HuggingFaceH4/MATH-500",
            "DigitalLearningGmbH/MATH-lighteval",
            "HuggingFaceH4/aime_2024",
        ],
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--system_prompt_type", default="normal")
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)

    # base vs MIRAGE-2
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["base", "mirage"],
        default="base",
        help="base: LLaDA default remasking, mirage: MIRAGE-2 (remasking='mirage-2')",
    )

    parser.add_argument(
        "--max_problems",
        type=int,
        default=-1,
        help="If >0, limit the number of problems evaluated.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print(f"[INFO] Sampler: {args.sampler}")

    model, tokenizer = build_model_and_tokenizer(args.model_path, device)
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("model.config.mask_token_id not found")

    if args.sampler == "base":
        remasking = "low_confidence"
    else:
        remasking = "mirage-2"  # MIRAGE 구현이 들어있는 모드

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_problems > 0:
        dataset = dataset.select(range(min(args.max_problems, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_math(
            b, tokenizer, system_prompt_type=args.system_prompt_type
        ),
    )

    extract_config = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                equations=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]

    total = 0
    num_correct = 0

    for batch in tqdm(dataloader, desc="Evaluating MATH-500"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        problems = batch["problems"]
        solutions = batch["solutions"]

        with torch.no_grad():
            outputs = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
            )

        gen_tokens = outputs[:, input_ids.shape[1] :]
        texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for prob, sol, ans in zip(problems, solutions, texts):
            total += 1

            # 정답 파싱
            gold_parsed = parse_solution(sol, extract_config)
            # 모델 답 파싱
            answer_parsed = parse(
                ans,
                extraction_config=extract_config,
            )

            is_correct = False
            if len(gold_parsed) > 0 and len(answer_parsed) > 0:
                try:
                    is_correct = bool(verify(gold_parsed, answer_parsed))
                except Exception:
                    is_correct = False

            if is_correct:
                num_correct += 1

    acc = num_correct / max(1, total) * 100.0
    print("====================================")
    print(f"Dataset      : {args.dataset_name}")
    print(f"Model        : {args.model_path}")
    print(f"Sampler      : {args.sampler} (remasking = {remasking})")
    print(f"Total problems: {total}")
    print(f"Accuracy      : {num_correct}/{total} = {acc:.2f}%")


if __name__ == "__main__":
    main()
