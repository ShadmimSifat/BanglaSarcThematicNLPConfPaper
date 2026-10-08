"""
BanglaSarcThematic: full implementation of the paper's pipeline.

Approaches
  direct    : single 5-class classifier
  stage1    : binary sarcasm vs non-sarcasm (all data)
  stage2    : 4-way thematic classifier (sarcastic samples only), standalone
  pipeline  : stage1 -> stage2 cascade, evaluated end-to-end on the 5-class test set
  all       : direct + stage1 + stage2 for every model, then the best pipeline

Usage examples
  python bangla_sarc_thematic.py --mode direct --model banglabert
  python bangla_sarc_thematic.py --mode stage1 --model banglabert
  python bangla_sarc_thematic.py --mode stage2 --model bge_m3
  python bangla_sarc_thematic.py --mode pipeline --stage1_model banglabert --stage2_model bge_m3
  python bangla_sarc_thematic.py --mode all
"""
import argparse
import json
import os
import random
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# Same ids as the notebook.
CLASS_LABELS = {0: "General", 1: "No Sarcasm", 2: "Hate Speech", 3: "GeoPolitics", 4: "Sports/Entertainment"}
NON_SARCASM_ID = 1
SARCASM_IDS = [0, 2, 3, 4]                        # stage-2 class k <-> 5-class id SARCASM_IDS[k]
STAGE2_NAMES = [CLASS_LABELS[i] for i in SARCASM_IDS]

# Accepts the notebook's names and the paper's names.
LABEL_ALIASES = {
    "general": 0, "non-thematic": 0, "non thematic": 0, "non-thematic sarcasm": 0, "nonthematic": 0,
    "no sarcasm": 1, "non-sarcasm": 1, "non sarcasm": 1, "non-sarcastic": 1, "not sarcasm": 1,
    "hate speech": 2, "hate expression": 2, "sarcastic hate expression": 2, "hate": 2,
    "geopolitics": 3, "geopolitical": 3, "geopolitical sarcasm": 3,
    "sports/entertainment": 4, "entertainment": 4, "entertainment sarcasm": 4, "sports": 4,
}

# name -> (HF id, pooling). Contrastive/embedding models use first token as in the paper,
# except Qwen3-Embedding, a decoder model whose sentence embedding is the last token.
MODELS = {
    "bangla_bert_base": ("sagorsarker/bangla-bert-base", "cls"),
    "banglabert": ("csebuetnlp/banglabert", "cls"),
    "xlm_roberta": ("xlm-roberta-base", "cls"),
    "bge_m3": ("BAAI/bge-m3", "cls"),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cls"),
    "qwen3_embedding": ("Qwen/Qwen3-Embedding-0.6B", "last"),
}

try:  # BanglaBERT's specialised normalizer: pip install git+https://github.com/csebuetnlp/normalizer
    from normalizer import normalize as bn_normalize
except Exception:
    bn_normalize = None

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF"
    "\U00002190-\U000021FF\U0000FE0F\U0000200D]+"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def preprocess(text, model_key):
    text = unicodedata.normalize("NFC", str(text))
    text = URL_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if model_key == "banglabert" and bn_normalize is not None:
        text = bn_normalize(text)
    return text


def to_label_id(x):
    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and float(x).is_integer()):
        return int(x)
    key = str(x).strip().lower()
    if key.isdigit():
        return int(key)
    if key not in LABEL_ALIASES:
        raise ValueError(f"Unknown label '{x}'. Add it to LABEL_ALIASES.")
    return LABEL_ALIASES[key]


def load_splits(args):
    """80/10/10 stratified split on the 5-class label. Cached so every experiment uses identical splits."""
    split_path = os.path.join(args.out_dir, "splits.csv")
    if os.path.exists(split_path):
        return pd.read_csv(split_path)
    df = pd.read_excel(args.data) if args.data.endswith((".xlsx", ".xls")) else pd.read_csv(args.data)
    df = df[[args.text_col, args.label_col]].dropna().rename(columns={args.text_col: "text", args.label_col: "label"})
    df["label"] = df["label"].map(to_label_id)
    df = df.reset_index(drop=True)
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=args.seed)
    tr, va, te = tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")
    out = pd.concat([tr, va, te]).reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out.to_csv(split_path, index=False)
    print({s: len(g) for s, g in out.groupby("split")})
    return out


def task_frame(df, task):
    """Map 5-class labels to the label space of the requested task."""
    df = df.copy()
    if task == "direct":
        df["y"] = df["label"]
    elif task == "stage1":                      # 1 = sarcasm, 0 = non-sarcasm
        df["y"] = (df["label"] != NON_SARCASM_ID).astype(int)
    elif task == "stage2":                      # sarcastic only, 0..3
        df = df[df["label"] != NON_SARCASM_ID]
        df["y"] = df["label"].map({c: i for i, c in enumerate(SARCASM_IDS)})
    return df.reset_index(drop=True)


TASK_CLASSES = {"direct": 5, "stage1": 2, "stage2": 4}


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts, self.labels = list(texts), labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], (-1 if self.labels is None else int(self.labels[i]))


def make_collate(tok, max_len, add_eos):
    def collate(batch):
        texts, ys = zip(*batch)
        if add_eos:
            texts = [t + tok.eos_token for t in texts]
        enc = tok(list(texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(ys, dtype=torch.long)
    return collate


# ----------------------------------------------------------------------------
# Model: encoder -> pooled token -> dropout(0.1) -> linear -> (softmax via CE / argmax)
# ----------------------------------------------------------------------------
class SarcasmClassifier(nn.Module):
    def __init__(self, hf_id, n_labels, pooling):
        super().__init__()
        self.enc = AutoModel.from_pretrained(hf_id)
        self.pooling = pooling
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(self.enc.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            x = hs[:, 0]
        else:  # last non-pad token (right padding)
            idx = attention_mask.sum(1) - 1
            x = hs[torch.arange(hs.size(0), device=hs.device), idx]
        return self.fc(self.drop(x.float()))


def build(model_key, n_labels):
    hf_id, pooling = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok, SarcasmClassifier(hf_id, n_labels, pooling), pooling == "last"


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------
def metrics_from(y_true, y_pred, n_labels, names=None):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred), "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "report": classification_report(y_true, y_pred, labels=list(range(n_labels)),
                                        target_names=names, zero_division=0, output_dict=True),
    }


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    logits, ys, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for ids, mask, y in loader:
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(ids, mask)
        if (y >= 0).all():
            loss_sum += ce(out.float().cpu(), y).item()
            n += len(y)
        logits.append(out.float().cpu())
        ys.append(y)
    logits = torch.cat(logits)
    return logits, torch.cat(ys).numpy(), (loss_sum / max(n, 1))


def run_task(task, model_key, splits, args):
    """Train one model on one task; return result dict and keep best checkpoint (by val loss)."""
    tag = f"__n{args.max_train_samples}" if args.max_train_samples else ""
    run_dir = os.path.join(args.out_dir, f"{task}__{model_key}{tag}")
    res_path = os.path.join(run_dir, "results.json")
    if os.path.exists(res_path) and not args.force:
        print(f"[skip] {task}/{model_key} already done")
        return json.load(open(res_path))
    os.makedirs(run_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp and torch.cuda.get_device_capability()[0] >= 8  # bf16 needs Ampere+; T4 falls back to fp32

    n_labels = TASK_CLASSES[task]
    data = {s: task_frame(splits[splits.split == s], task) for s in ("train", "val", "test")}
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.max_train_samples:  # quick/smoke runs only: results are NOT comparable to the paper
        n_eval = args.max_eval_samples or max(args.max_train_samples // 4, 32)
        data["train"] = data["train"].sample(min(args.max_train_samples, len(data["train"])), random_state=args.seed).reset_index(drop=True)
        for s_ in ("val", "test"):
            data[s_] = data[s_].sample(min(n_eval, len(data[s_])), random_state=args.seed).reset_index(drop=True)
        print(f"[subset] train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}")
    for s in data:
        data[s]["text"] = data[s]["text"].map(lambda t: preprocess(t, model_key))

    tok, model, add_eos = build(model_key, n_labels)
    model.to(device)
    if args.grad_ckpt:
        model.enc.gradient_checkpointing_enable()
    if args.freeze_embeddings:  # big speed-up on CPU: the 250k-word embedding matrix is most of the parameters
        for p_ in model.enc.get_input_embeddings().parameters():
            p_.requires_grad = False
    collate = make_collate(tok, args.max_len, add_eos)
    loaders = {
        s: DataLoader(TextDataset(d["text"], d["y"].values), batch_size=args.batch_size,
                      shuffle=(s == "train"), collate_fn=collate, num_workers=0)
        for s, d in data.items()
    }

    weights = None
    if args.class_weights:  # optional (used in the authors' notebook, not stated in the paper)
        cnt = np.bincount(data["train"]["y"], minlength=n_labels)
        weights = torch.tensor(len(data["train"]) / (n_labels * np.maximum(cnt, 1)), dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)], "weight_decay": args.wd},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)

    import time
    best_val, best_epoch, bad, history = float("inf"), -1, 0, []
    ckpt = os.path.join(run_dir, "best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, steps, t0 = 0.0, 0, time.time()
        n_steps = len(loaders["train"])
        for ids, mask, y in loaders["train"]:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(ids, mask)
            loss = loss_fn(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            steps += 1
            if steps == 1 or steps % args.log_every == 0:
                el = time.time() - t0
                print(f"  step {steps}/{n_steps} loss {tot/steps:.4f} | {el:.0f}s elapsed, ~{el/steps*(n_steps-steps):.0f}s left in epoch", flush=True)
        v_logits, v_y, v_loss = predict(model, loaders["val"], device, amp)
        v_acc = accuracy_score(v_y, v_logits.argmax(1).numpy())
        history.append({"epoch": epoch, "train_loss": tot / steps, "val_loss": v_loss, "val_acc": v_acc})
        print(f"[{task}/{model_key}] epoch {epoch:02d} train {tot/steps:.4f} val {v_loss:.4f} acc {v_acc:.4f}")
        if v_loss < best_val:
            best_val, best_epoch, bad = v_loss, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    t_logits, t_y, _ = predict(model, loaders["test"], device, amp)
    v_logits, v_y, _ = predict(model, loaders["val"], device, amp)
    names = {"direct": [CLASS_LABELS[i] for i in range(5)], "stage1": ["Non-Sarcasm", "Sarcasm"], "stage2": STAGE2_NAMES}[task]
    result = {
        "task": task, "model": model_key, "best_epoch": best_epoch, "history": history,
        "val_accuracy": accuracy_score(v_y, v_logits.argmax(1).numpy()),
        "test": metrics_from(t_y, t_logits.argmax(1).numpy(), n_labels, names),
    }
    json.dump(result, open(res_path, "w"), indent=2)
    tok.save_pretrained(run_dir)
    print(f"[{task}/{model_key}] TEST acc {result['test']['accuracy']:.4f} macro-F1 {result['test']['macro_f1']:.4f}")
    del model, opt
    torch.cuda.empty_cache()
    return result


# ----------------------------------------------------------------------------
# Hierarchical end-to-end pipeline
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_texts(task, model_key, texts, args):
    run_dir = os.path.join(args.out_dir, f"{task}__{model_key}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp and torch.cuda.get_device_capability()[0] >= 8  # bf16 needs Ampere+; T4 falls back to fp32
    tok, model, add_eos = build(model_key, TASK_CLASSES[task])
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.to(device)
    texts = [preprocess(t, model_key) for t in texts]
    loader = DataLoader(TextDataset(texts), batch_size=args.batch_size, collate_fn=make_collate(tok, args.max_len, add_eos))
    logits, _, _ = predict(model, loader, device, amp)
    del model
    torch.cuda.empty_cache()
    return logits.argmax(1).numpy()


def run_pipeline(stage1_model, stage2_model, splits, args):
    test = splits[splits.split == "test"].reset_index(drop=True)
    s1 = predict_texts("stage1", stage1_model, test["text"], args)       # 1 = sarcasm
    final = np.full(len(test), NON_SARCASM_ID)
    sarc_idx = np.where(s1 == 1)[0]
    if len(sarc_idx):
        s2 = predict_texts("stage2", stage2_model, test["text"].iloc[sarc_idx], args)
        final[sarc_idx] = [SARCASM_IDS[k] for k in s2]                   # error propagation preserved
    m = metrics_from(test["label"].values, final, 5, [CLASS_LABELS[i] for i in range(5)])
    result = {"stage1_model": stage1_model, "stage2_model": stage2_model, "test": m}
    out = os.path.join(args.out_dir, f"pipeline__{stage1_model}__{stage2_model}.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"[pipeline {stage1_model}+{stage2_model}] acc {m['accuracy']:.4f} P {m['macro_precision']:.4f} "
          f"R {m['macro_recall']:.4f} F1 {m['macro_f1']:.4f}")
    return result


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "stage1", "stage2", "pipeline", "all"], default="all")
    ap.add_argument("--model", choices=list(MODELS), help="model for direct/stage1/stage2 (default: all six)")
    ap.add_argument("--stage1_model", choices=list(MODELS), default="banglabert")
    ap.add_argument("--stage2_model", choices=list(MODELS), default="bge_m3")
    ap.add_argument("--data", default="dataset/dataset2.xlsx")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--max_len", type=int, default=512)       # paper: 512
    ap.add_argument("--batch_size", type=int, default=16)     # paper: 16
    ap.add_argument("--lr", type=float, default=2e-5)         # paper: 2e-5
    ap.add_argument("--wd", type=float, default=0.01)         # paper: 0.01
    ap.add_argument("--epochs", type=int, default=50)         # upper bound; early stopping decides
    ap.add_argument("--patience", type=int, default=15)       # paper: 15
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weights", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true", help="save GPU memory on the big models")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--max_train_samples", type=int, default=0, help="train on a small subset (smoke tests / weak CPUs)")
    ap.add_argument("--max_eval_samples", type=int, default=0, help="val/test subset size (default: train subset / 4)")
    ap.add_argument("--freeze_embeddings", action="store_true", help="freeze token embeddings (much faster on CPU)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = default)")
    ap.add_argument("--log_every", type=int, default=20, help="print progress every N steps")
    ap.add_argument("--force", action="store_true", help="retrain even if results exist")
    args = ap.parse_args()

    splits = load_splits(args)
    keys = [args.model] if args.model else list(MODELS)

    if args.mode in ("direct", "stage1", "stage2"):
        for k in keys:
            run_task(args.mode, k, splits, args)
    elif args.mode == "pipeline":
        run_pipeline(args.stage1_model, args.stage2_model, splits, args)
    else:  # all
        res = {t: {k: run_task(t, k, splits, args) for k in keys} for t in ("direct", "stage1", "stage2")}
        # choose stage models by VALIDATION accuracy (avoids test-set selection)
        best1 = max(res["stage1"], key=lambda k: res["stage1"][k]["val_accuracy"])
        best2 = max(res["stage2"], key=lambda k: res["stage2"][k]["val_accuracy"])
        pipe = run_pipeline(best1, best2, splits, args)
        rows = [{"task": t, "model": k, "acc": r["test"]["accuracy"], "P": r["test"]["macro_precision"],
                 "R": r["test"]["macro_recall"], "F1": r["test"]["macro_f1"]}
                for t, d in res.items() for k, r in d.items()]
        rows.append({"task": "pipeline", "model": f"{best1}+{best2}", "acc": pipe["test"]["accuracy"],
                     "P": pipe["test"]["macro_precision"], "R": pipe["test"]["macro_recall"], "F1": pipe["test"]["macro_f1"]})
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
