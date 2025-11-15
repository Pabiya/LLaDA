import os
import argparse
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from generate import generate  # 네가 MIRAGE 구현해 둔 generate.py
from eval.gsm8k import GSM8KDataset
from eval.countdown import CTDDataset
from eval.sudoku import SudokuDataset
from parsers import Parser, is_equiv


DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["gsm8k", "countdown", "sudoku"],
        default="sudoku",
    )
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
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
        "--subsample",
        type=int,
        default=-1,
        help="If >0, subsample that many examples from the dataset.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print(f"[INFO] Sampler: {args.sampler} (base vs MIRAGE)")

    model, tokenizer = build_model_and_tokenizer(args.model_path, device)
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("model.config.mask_token_id not found")

    # remasking 모드 매핑
    if args.sampler == "base":
        remasking = "low_confidence"   # LLaDA 원래 설정
    else:
        remasking = "mirage-2"         # 네가 구현한 MIRAGE-2 모드

    DatasetCls = DATASET_MAP[args.dataset_name]

    ds = DatasetCls(
        tokenizer,
        subsample=args.subsample,
        num_examples=0,      # few-shot 안 씀
        add_reasoning=False, # prefill 없음
    )

    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ds.collate_fn,
    )

    total = 0
    num_correct = 0

    # sudoku는 per-cell accuracy도 계산
    sudoku_correct_cells = 0
    sudoku_total_cells = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        questions = batch["questions"]
        answers = batch["answers"]

        with torch.no_grad():
            # generate.py 의 generate()를 직접 사용
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

        # 프롬프트 이후 부분만 디코딩
        gen_tokens = outputs[:, input_ids.shape[1] :]
        texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for text, q, a in zip(texts, questions, answers):
            total += 1

            if args.dataset_name == "gsm8k":
                # GSM8K: 숫자만 뽑아서 is_equiv 으로 비교
                pred_ans = Parser.extract_answer_gsm8k(text)
                is_correct = is_equiv(pred_ans, a)
                if is_correct:
                    num_correct += 1

            elif args.dataset_name == "countdown":
                # Countdown: dataset의 validate 사용
                is_correct = ds.validate(text, a, question=q)
                if is_correct:
                    num_correct += 1

            elif args.dataset_name == "sudoku":
                # Sudoku: validate가 (correct_cells, total_cells, acc) 또는 (cells, tot, acc) 를 반환
                correct_cells, total_cells, acc = ds.validate(text, a, question=q)
                sudoku_correct_cells += correct_cells
                sudoku_total_cells += total_cells
                if acc == 1.0:
                    num_correct += 1

    acc = num_correct / max(1, total) * 100.0
    print("====================================")
    print(f"Dataset      : {args.dataset_name}")
    print(f"Model        : {args.model_path}")
    print(f"Sampler      : {args.sampler}  (remasking = {remasking})")
    print(f"Total samples: {total}")
    print(f"Accuracy     : {num_correct}/{total} = {acc:.2f}%")

    if args.dataset_name == "sudoku" and sudoku_total_cells > 0:
        cell_acc = sudoku_correct_cells / sudoku_total_cells * 100.0
        print(f"Sudoku cell-level accuracy: "
              f"{sudoku_correct_cells}/{sudoku_total_cells} = {cell_acc:.2f}%")


if __name__ == "__main__":
    main()
