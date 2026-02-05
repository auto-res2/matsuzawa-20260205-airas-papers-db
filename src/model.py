import math
import random
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from src.preprocess import extract_numbers, normalize_number_str


def hoeffding_rad(n, delta):
    return math.sqrt(math.log(2.0 / delta) / (2.0 * max(1, n)))


def beta_q(a, b, q):
    dist = torch.distributions.Beta(torch.tensor(float(a)), torch.tensor(float(b)))
    return float(dist.icdf(torch.tensor(float(q))))


class GenModelWrapper:
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.runs.model.name, cache_dir=".cache/")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        dtype = torch.bfloat16 if cfg.runs.model.dtype == "bf16" else torch.float32
        if "t5" in cfg.runs.model.name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.runs.model.name,
                cache_dir=".cache/",
                torch_dtype=dtype,
                device_map=cfg.runs.model.device_map,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.runs.model.name,
                cache_dir=".cache/",
                torch_dtype=dtype,
                device_map=cfg.runs.model.device_map,
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def generate(self, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, num_return_sequences: int = 1):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if isinstance(self.model, AutoModelForCausalLM) or self.model.config.is_decoder:
            input_len = inputs["input_ids"].shape[1]
            outs = outs[:, input_len:]
        return [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outs]


class ProxyCalibrator:
    def __init__(self, a_pi=1, b_pi=1, a_nu=1, b_nu=1):
        self.a_pi, self.b_pi = a_pi, b_pi
        self.a_nu, self.b_nu = a_nu, b_nu

    def update(self, proxy_outcome: int, true_outcome: int):
        if proxy_outcome == 1:
            if true_outcome == 1:
                self.a_pi += 1
            else:
                self.b_pi += 1
        else:
            if true_outcome == 0:
                self.a_nu += 1
            else:
                self.b_nu += 1

    def bounds(self, alpha=0.05):
        l_pi = beta_q(self.a_pi, self.b_pi, alpha / 2)
        u_pi = beta_q(self.a_pi, self.b_pi, 1 - alpha / 2)
        l_nu = beta_q(self.a_nu, self.b_nu, alpha / 2)
        u_nu = beta_q(self.a_nu, self.b_nu, 1 - alpha / 2)
        return (l_pi, u_pi, l_nu, u_nu)

    def mean_pi(self):
        return self.a_pi / (self.a_pi + self.b_pi)

    def mean_nu(self):
        return self.a_nu / (self.a_nu + self.b_nu)


def direct_generate(model: GenModelWrapper, x: str, gen_cfg) -> str:
    prompt = f"Solve and output ONLY the number.\nProblem: {x}\nAnswer:"
    return model.generate(prompt, gen_cfg.max_new_tokens, False, gen_cfg.temperature)[0]


def self_refine_generate(model: GenModelWrapper, x: str, gen_cfg, T=3) -> Tuple[str, Dict[str, int]]:
    y = direct_generate(model, x, gen_cfg)
    for _ in range(T):
        fb = model.generate(
            "You are a strict math checker. Given the problem and the proposed answer, say what is wrong (if anything) and how to fix it.\n"
            f"Problem: {x}\nProposed answer: {y}\nFeedback:",
            gen_cfg.max_new_tokens,
            False,
            gen_cfg.temperature,
        )[0]
        y = model.generate(
            "Revise the answer using the feedback.\nIMPORTANT: Put the final answer first, then you may add a brief explanation.\n"
            f"Problem: {x}\nCurrent answer: {y}\nFeedback: {fb}\nRevised response:",
            gen_cfg.max_new_tokens,
            True,
            gen_cfg.temperature,
        )[0]
    return y, {"proxy_calls": 0, "audit_calls": 0, "rounds": T}


def proxy_score_lenient(pred_text: str, gold: str, p_flip: float, rng: random.Random) -> float:
    nums = extract_numbers(pred_text)
    s = 1.0 if normalize_number_str(gold) in [normalize_number_str(n) for n in nums] else 0.0
    if rng.random() < p_flip:
        s = 1.0 - s
    return s


def true_score_strict(pred_text: str, gold: str) -> float:
    nums = extract_numbers(pred_text)
    if len(nums) == 0:
        return 0.0
    return 1.0 if normalize_number_str(nums[0]) == normalize_number_str(gold) else 0.0


def scg_sr_proxy(model: GenModelWrapper, x: str, gold: str, cfg, p_flip: float, lam: float, tau: float, per_round_proxy_budget: int):
    rng = random.Random(cfg.seed)
    T = cfg.method_cfg.T
    B = cfg.method_cfg.B
    delta = cfg.method_cfg.delta
    patience = cfg.method_cfg.patience

    def util(mean_proxy, y_text):
        return mean_proxy - lam * len(model.tokenize(y_text))

    y_star = direct_generate(model, x, cfg.generation)
    n_star, sum_star = 1, proxy_score_lenient(y_star, gold, p_flip, rng)
    proxy_calls = 1
    audit_calls = 0
    bad = 0

    for t in range(T):
        fb = model.generate(
            "You are a strict math checker. Given the problem and the proposed answer, say what is wrong (if anything) and how to fix it.\n"
            f"Problem: {x}\nProposed answer: {y_star}\nFeedback:",
            cfg.generation.max_new_tokens,
            False,
            cfg.generation.temperature,
        )[0]
        refine_prompt = (
            "Revise the answer using the feedback.\nIMPORTANT: Put the final answer first, then you may add a brief explanation.\n"
            f"Problem: {x}\nCurrent answer: {y_star}\nFeedback: {fb}\nRevised response:"
        )
        cands = model.generate(refine_prompt, cfg.generation.max_new_tokens, True, cfg.generation.temperature, num_return_sequences=B)

        n = [0] * B
        s = [0.0] * B
        evals = 0
        for i in range(B):
            if evals >= per_round_proxy_budget:
                break
            n[i] += 1
            s[i] += proxy_score_lenient(cands[i], gold, p_flip, rng)
            evals += 1
            proxy_calls += 1
        if evals < per_round_proxy_budget:
            n_star += 1
            sum_star += proxy_score_lenient(y_star, gold, p_flip, rng)
            evals += 1
            proxy_calls += 1

        while evals < per_round_proxy_budget:
            ucbs = []
            lcbs = []
            for i in range(B):
                mean_i = s[i] / max(1, n[i])
                r = hoeffding_rad(n[i], delta / (B + 1))
                ucbs.append(util(min(1.0, mean_i + r), cands[i]))
                lcbs.append(util(max(0.0, mean_i - r), cands[i]))
            order = sorted(range(B), key=lambda i: ucbs[i], reverse=True)
            best_i = order[0]
            second_i = order[1] if B > 1 else order[0]
            if lcbs[best_i] > ucbs[second_i] + 1e-9:
                break
            i = best_i
            n[i] += 1
            s[i] += proxy_score_lenient(cands[i], gold, p_flip, rng)
            evals += 1
            proxy_calls += 1

        best_i, best_lcb = 0, -1e9
        for i in range(B):
            mean_i = s[i] / max(1, n[i])
            r = hoeffding_rad(n[i], delta / (B + 1))
            lcb = util(max(0.0, mean_i - r), cands[i])
            if lcb > best_lcb:
                best_lcb, best_i = lcb, i

        mean_star = sum_star / n_star
        r_star = hoeffding_rad(n_star, delta / (B + 1))
        ucb_star = util(min(1.0, mean_star + r_star), y_star)

        if best_lcb > ucb_star + tau:
            y_star = cands[best_i]
            n_star, sum_star = 1, proxy_score_lenient(y_star, gold, p_flip, rng)
            proxy_calls += 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                return y_star, {"proxy_calls": proxy_calls, "audit_calls": audit_calls, "rounds": t + 1}

    return y_star, {"proxy_calls": proxy_calls, "audit_calls": audit_calls, "rounds": T}


def accr_refine(
    model: GenModelWrapper,
    x: str,
    gold: str,
    cfg,
    calibrator: ProxyCalibrator,
    p_flip: float,
    lam: float,
    tau: float,
    per_round_audit_budget: int,
    alpha_cal: float,
):
    rng = random.Random(cfg.seed)
    T = cfg.method_cfg.T
    B = cfg.method_cfg.B
    delta = cfg.method_cfg.delta
    per_round_proxy_budget = cfg.method_cfg.per_round_proxy_budget
    patience = cfg.method_cfg.patience

    def util_from_trueprob(p_true, y_text):
        return p_true - lam * len(model.tokenize(y_text))

    def calibrated_bounds_from_proxy(mean_proxy, y_text):
        l_pi, u_pi, l_nu, u_nu = calibrator.bounds(alpha=alpha_cal)
        l_true = mean_proxy * l_pi + (1.0 - mean_proxy) * (1.0 - u_nu)
        u_true = mean_proxy * u_pi + (1.0 - mean_proxy) * (1.0 - l_nu)
        return util_from_trueprob(l_true, y_text), util_from_trueprob(u_true, y_text)

    y_star = direct_generate(model, x, cfg.generation)
    n_star, sum_star = 1, proxy_score_lenient(y_star, gold, p_flip, rng)
    proxy_calls = 1
    audit_calls = 0
    bad = 0

    for t in range(T):
        fb = model.generate(
            "You are a strict math checker. Given the problem and the proposed answer, say what is wrong (if anything) and how to fix it.\n"
            f"Problem: {x}\nProposed answer: {y_star}\nFeedback:",
            cfg.generation.max_new_tokens,
            False,
            cfg.generation.temperature,
        )[0]
        refine_prompt = (
            "Revise the answer using the feedback.\nIMPORTANT: Put the final answer first, then you may add a brief explanation.\n"
            f"Problem: {x}\nCurrent answer: {y_star}\nFeedback: {fb}\nRevised response:"
        )
        cands = model.generate(refine_prompt, cfg.generation.max_new_tokens, True, cfg.generation.temperature, num_return_sequences=B)

        n = [0] * B
        s = [0.0] * B
        proxy_first = [None] * B

        evals = 0
        for i in range(B):
            if evals >= per_round_proxy_budget:
                break
            out = proxy_score_lenient(cands[i], gold, p_flip, rng)
            proxy_first[i] = int(out)
            n[i] += 1
            s[i] += out
            evals += 1
            proxy_calls += 1
        if evals < per_round_proxy_budget:
            n_star += 1
            sum_star += proxy_score_lenient(y_star, gold, p_flip, rng)
            evals += 1
            proxy_calls += 1

        while evals < per_round_proxy_budget:
            ucbs = []
            for i in range(B):
                mean_i = s[i] / max(1, n[i])
                r = hoeffding_rad(n[i], delta / (B + 1))
                ucbs.append(min(1.0, mean_i + r))
            i = max(range(B), key=lambda j: ucbs[j])
            n[i] += 1
            s[i] += proxy_score_lenient(cands[i], gold, p_flip, rng)
            evals += 1
            proxy_calls += 1

        means = [s[i] / max(1, n[i]) for i in range(B)]
        audit_order = sorted(range(B), key=lambda i: means[i], reverse=True)
        for k in range(min(per_round_audit_budget, B)):
            i = audit_order[k]
            p0 = proxy_first[i]
            if p0 is None:
                p0 = int(means[i] >= 0.5)
            t_true = int(true_score_strict(cands[i], gold))
            calibrator.update(p0, t_true)
            audit_calls += 1

        best_i, best_lcbU = 0, -1e9
        for i in range(B):
            lcbU, _ = calibrated_bounds_from_proxy(means[i], cands[i])
            if lcbU > best_lcbU:
                best_lcbU, best_i = lcbU, i

        mean_star = sum_star / n_star
        _, ucbU_star = calibrated_bounds_from_proxy(mean_star, y_star)

        if best_lcbU > ucbU_star + tau:
            y_star = cands[best_i]
            n_star, sum_star = 1, proxy_score_lenient(y_star, gold, p_flip, rng)
            proxy_calls += 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                return y_star, {"proxy_calls": proxy_calls, "audit_calls": audit_calls, "rounds": t + 1}

    return y_star, {"proxy_calls": proxy_calls, "audit_calls": audit_calls, "rounds": T}
