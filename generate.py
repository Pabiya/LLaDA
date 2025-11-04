import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

# @ torch.no_grad()
# def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#         logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
#         confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
#     '''
#     x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     if attention_mask is not None:
#         attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

#     prompt_index = (x != mask_id)

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length

#     assert steps % num_blocks == 0
#     steps = steps // num_blocks

#     for num_block in range(num_blocks):
#         block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#         for i in range(steps):
#             mask_index = (x == mask_id)
#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 if attention_mask is not None:
#                     attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
#                 logits = model(x_, attention_mask=attention_mask_).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x, attention_mask=attention_mask).logits

#             if logits_eos_inf:
#                 logits[:, :, 126081] = -torch.inf

#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
#             if confidence_eos_eot_inf:
#                 logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

#             if remasking == 'low_confidence':
#                 p = F.softmax(logits, dim=-1)
#                 x0_p = torch.squeeze(
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)

#             x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)

#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]

#     return x

@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # entropy cache (used by mirage-2); inf = "no record", so replaced by current entropy when needed
    H_cache = torch.full((x.shape[0], x.shape[1]), float("inf"), device=x.device, dtype=torch.float32)

    for num_block in range(num_blocks):
        # mask indexes inside the current block (for step-wise transfer counts)
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)  # [B, steps]

        for i in range(steps):
            mask_index = (x == mask_id)

            # classifier-free guidance forward
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            # optional: forbid EOS in logits
            if logits_eos_inf:
                # EOS id 126081; keep original behavior
                logits[:, :, 126081] = -torch.inf

            # 1st pass decoding with Gumbel trick (or identity if temperature==0)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

            # optional: forbid EOS/EoT in confidence tensor path (kept for backward-compat)
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            # CURRENT probability & entropy (used by mirage branches)
            p_cur = F.softmax(logits.to(torch.float32), dim=-1)                  # [B, L, V]
            # If you want to exclude [MASK] prob and renormalize, uncomment:
            p_cur[..., mask_id] = 0
            p_cur = p_cur / p_cur.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            
            H_cur = -(p_cur.clamp_min(1e-12) * p_cur.clamp_min(1e-12).log()).sum(dim=-1)  # [B, L]

            if remasking == 'low_confidence':
                p = p_cur  # already computed
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking in ('mirage-1', 'mirage-2'):
                x0_p = None  # not used in MIRAGE branches
            else:
                raise NotImplementedError(remasking)

            # exclude positions beyond current block for selection
            # For original branches, keep behavior:
            if remasking in ('low_confidence',):
                x0_p[:, block_end:] = -np.inf

            # Stitch: for masked positions we propose x0; otherwise keep x
            x0 = torch.where(mask_index, x0, x)

            # ORIGINAL low_confidence/random UPDATE
            if remasking in ('low_confidence', 'random'):
                # only masked positions are candidates; unmasked = -inf
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

                # update entropy cache for newly unmasked (optional: useful if later switching to mirage-2 in same run)
                newly_unmasked = transfer_index  # from mask -> token this step
                H_cache[newly_unmasked] = H_cur[newly_unmasked]

            elif remasking == 'low_confidence_all':
                B, Ltot = x.shape

                # 후보 영역: 프롬프트 제외 + 현재 블록 끝까지(= 지금까지 생성한 모든 토큰)
                block_start = prompt.shape[1] + num_block * block_length
                block_end   = prompt.shape[1] + (num_block + 1) * block_length

                upto_current = torch.zeros_like(mask_index)
                upto_current[:, :block_end] = True
                non_prompt = ~prompt_index
                cand_mask = upto_current & non_prompt  # 우리가 재조정(keep/remask)할 수 있는 모든 위치

                # 현재 step 분포 (이미 계산되어 있음): p_cur = softmax(logits)
                # confidence for "masked→token" 후보는 x0의 확률로, 
                # "이미 언마스크된" 후보는 현재 토큰 x의 확률로 점수화한다.
                #   S_masked = p_cur[b,l, x0[b,l]]
                #   S_prev   = p_cur[b,l, x[b,l]]
                S_conf = torch.full((B, Ltot), -float("inf"), device=x.device, dtype=torch.float32)

                # (1) masked 후보 점수
                masked_cands = (mask_index & cand_mask)
                if masked_cands.any():
                    S_conf[masked_cands] = torch.gather(
                        p_cur[masked_cands.unsqueeze(-1).expand(-1,-1,p_cur.shape[-1])],
                        dim=-1,
                        index=x0[masked_cands].unsqueeze(-1)
                    ).squeeze(-1)

                # (2) 이미 언마스크된 후보 점수
                prev_cands = ((~mask_index) & cand_mask)
                if prev_cands.any():
                    S_conf[prev_cands] = torch.gather(
                        p_cur[prev_cands.unsqueeze(-1).expand(-1,-1,p_cur.shape[-1])],
                        dim=-1,
                        index=x[prev_cands].unsqueeze(-1)
                    ).squeeze(-1)

                # 스케줄 보존: 이번 step의 최종 언마스크 수 = (이전 언마스크 수) + num_transfer_tokens
                keep_index = torch.zeros_like(mask_index)

                # 이전 mask 상태를 저장(이 step에서 새로 unmask 된 위치를 알아내기 위함)
                was_mask = mask_index.clone()

                for j in range(B):
                    # 현재 후보 영역에서 이전 언마스크 위치들
                    prev_unmasked_idx = ((~mask_index[j]) & cand_mask[j]).nonzero(as_tuple=False).squeeze(-1)
                    prev_unmasked_cnt = int(prev_unmasked_idx.numel())

                    k_unmask = int(num_transfer_tokens[j, i].item())  # 이번 step에 mask→token으로 전이해야 할 개수(스케줄)

                    # (A) masked 중에서 "확신 높은(top confidence)" k_unmask개를 먼저 선택해 이번 step에 언마스크로 전환
                    masked_idx = (mask_index[j] & cand_mask[j]).nonzero(as_tuple=False).squeeze(-1)
                    chosen_masked = masked_idx.new_empty((0,), dtype=torch.long)
                    if masked_idx.numel() > 0 and k_unmask > 0:
                        conf_masked = S_conf[j, masked_idx]
                        k_sel = min(k_unmask, masked_idx.numel())
                        _, ord_masked = torch.topk(conf_masked, k=k_sel)  # 높은 확률 우선
                        chosen_masked = masked_idx[ord_masked]

                    keep_set = set(chosen_masked.tolist())

                    # (B) 나머지는 이전 언마스크들 중에서 선택해 "총 keep 수 = prev_unmasked_cnt + k_unmask"가 되도록
                    target_keep = prev_unmasked_cnt + k_unmask
                    need_prev_keep = target_keep - len(keep_set)

                    if need_prev_keep > 0 and prev_unmasked_idx.numel() > 0:
                        conf_prev = S_conf[j, prev_unmasked_idx]
                        k_prev = min(need_prev_keep, prev_unmasked_idx.numel())
                        _, ord_prev = torch.topk(conf_prev, k=k_prev)  # 높은 확률 우선 keep
                        chosen_prev = prev_unmasked_idx[ord_prev]
                        keep_set.update(chosen_prev.tolist())

                    # 프롬프트는 항상 keep
                    keep_index[j, :prompt.shape[1]] = prompt_index[j, :prompt.shape[1]]
                    if len(keep_set) > 0:
                        keep_index[j, list(keep_set)] = True

                # 최종 적용:
                #  - (mask였고 keep으로 뽑힌) 위치는 이번 step 예측 x0로 채움
                newly_kept_from_mask = keep_index & was_mask
                x[newly_kept_from_mask] = x0[newly_kept_from_mask]

                #  - keep 아닌 (프롬프트 제외) 위치는 [MASK]로 되돌림 → 이전 언마스크들도 low-confidence면 remask됨
                drop_index = (~keep_index) & (~prompt_index)
                x[drop_index] = mask_id

            # MIRAGE BRANCHES
            elif remasking in ('mirage-1', 'mirage-2'):
                B, Ltot = x.shape

                # Candidate region for keep/remask decision:
                #   - Exclude prompt
                #   - Include everything up to CURRENT block_end
                #   → This allows remasking tokens from previous blocks as requested.
                upto_current = torch.zeros_like(mask_index)
                upto_current[:, :block_end] = True
                non_prompt = ~prompt_index
                cand_mask = upto_current & non_prompt  # candidates we may keep or mask

                # Build the scoring tensor H_use:
                #   mirage-1: use CURRENT entropy everywhere
                #   mirage-2: for positions that are ALREADY unmasked, prefer CACHED entropy at unmask time (if cached),
                #             otherwise fall back to CURRENT entropy; for masked positions, use CURRENT entropy.
                # We also set score = +inf outside candidate region so they are never selected as keep.
                H_use = torch.full_like(H_cur, float("inf"))
                H_use[cand_mask] = H_cur[cand_mask]  # default = current entropy in candidate region
                if remasking == 'mirage-2':
                    cur_unmasked = ((~mask_index) & cand_mask)
                    # use min(H_cache, H_cur) so that inf is replaced by current; otherwise prefer cached
                    H_use[cur_unmasked] = torch.minimum(H_cache[cur_unmasked], H_cur[cur_unmasked])

                # We now select the final "keep" set to PRESERVE SCHEDULE:
                # target_keep = (number of currently unmasked within candidates) + num_transfer_tokens
                keep_index = torch.zeros_like(mask_index)

                # helpful tensors for batch loop
                # positions that were masked before update (for "newly kept from mask")
                was_mask = mask_index.clone()

                for j in range(B):
                    # count previously unmasked within candidate region
                    prev_unmasked_idx = ((~mask_index[j]) & cand_mask[j]).nonzero(as_tuple=False).squeeze(-1)
                    prev_unmasked_cnt = int(prev_unmasked_idx.numel())

                    # exact number of mask->token transitions we must add this step (from schedule)
                    k_unmask = int(num_transfer_tokens[j, i].item())

                    # (1) choose k_unmask masked positions (within candidate region) with LOWEST entropy to unmask now
                    masked_candidates = (mask_index[j] & cand_mask[j]).nonzero(as_tuple=False).squeeze(-1)
                    chosen_masked = masked_candidates.new_empty((0,), dtype=torch.long)
                    if masked_candidates.numel() > 0 and k_unmask > 0:
                        H_masked = H_use[j, masked_candidates]  # current entropy (mirage-1/2 same for masked)
                        k_unmask = min(k_unmask, masked_candidates.numel())
                        # pick smallest H → use topk on -H
                        _, ord_idx = torch.topk(-H_masked, k=k_unmask)
                        chosen_masked = masked_candidates[ord_idx]

                    # current keep set starts with these new unmasked
                    keep_set = set(chosen_masked.tolist())

                    # (2) fill the rest from previously unmasked positions (lowest H first) to hit target_keep
                    target_keep = prev_unmasked_cnt + len(keep_set)
                    # Actually, we must reach exactly prev_unmasked_cnt + num_transfer_tokens (not +len(keep_set) which == num_transfer_tokens)
                    target_keep = prev_unmasked_cnt + int(num_transfer_tokens[j, i].item())

                    need_more = target_keep - len(keep_set)
                    if need_more > 0 and prev_unmasked_idx.numel() > 0:
                        H_prev = H_use[j, prev_unmasked_idx]  # mirage-1: current H; mirage-2: cached-or-current H
                        k_keep_prev = min(need_more, prev_unmasked_idx.numel())
                        _, ord2 = torch.topk(-H_prev, k=k_keep_prev)
                        chosen_prev = prev_unmasked_idx[ord2]
                        keep_set.update(chosen_prev.tolist())

                    # (3) ensure prompt is always kept
                    keep_index[j, :prompt.shape[1]] = prompt_index[j, :prompt.shape[1]]
                    if len(keep_set) > 0:
                        keep_index[j, list(keep_set)] = True

                # (4) Apply the keep/mask edits:
                # - For positions that were masked and are now kept → write new tokens x0 there
                newly_kept_from_mask = keep_index & was_mask
                x[newly_kept_from_mask] = x0[newly_kept_from_mask]

                # - For positions not kept (and not prompt) → set to [MASK]
                drop_index = (~keep_index) & (~prompt_index)
                x[drop_index] = mask_id

                # (5) update entropy cache for positions that became unmasked THIS step (mask->token)
                H_cache[newly_kept_from_mask] = H_cur[newly_kept_from_mask]

            # end of remasking branches
        # end of steps
    # end of blocks

    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
