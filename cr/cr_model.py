"""
cr_model.py - contradiction-only MOLDCAST for the SparseCL benchmarks.

Two encoders, as in SparseCL (frozen cosine model + a trained model for the contradiction signal):

  CosEncoder   a frozen off-the-shelf sentence embedder, pooled and normalised exactly as SparseCL's
               zero-shot setting (bge/gte: CLS token; others: mean), giving cos(q, d).
  SlotEncoder  a fine-tuned transformer + slot head. n slots per passage; slot c has a commitment
               a_c in (0, 1) and a value VECTOR v_c in R^k. The clash between query q and
               passage d is
                   K(q, d) = (1/n) sum_c a^q_c a^d_c ||v^d_c - v^q_c||^2
               (MOLDCAST's K; k = 1 gives the scalar form). Values are linear by default;
               --value_act tanh bounds them to (-2, 2) but lets training saturate them;
               --value_act sphere puts them on the unit sphere (bounded, no plateau: then
               K = (2/n) sum_c a^q_c a^d_c (1 - cos(v^q_c, v^d_c)), and stored vectors have nearly
               uniform length, which approximate inner-product search needs). Expanding the square makes K an exact
               inner product <query_vec(q), stored_vec(d)>, so every passage of the corpus is
               scored by one matrix product and searchable with any MIPS index:
                   stored(d) = [a^d ||v^d||^2 , a^d , a^d * v^d]          (n + n + n k numbers)
                   query(q)  = [a^q , a^q ||v^q||^2 , -2 a^q * v^q] / n

Topic gate (--gate_dim r > 0, the default): K grows with any dissimilarity, so unrelated passages
would out-clash the true contradiction. The clash is therefore gated by a learned topical
similarity, a small projection t = normalize(P u) of the frozen cosine embedding u:
                   GK(q, d) = <t_q, t_d> * K(q, d) = <t_q (x) query(q), t_d (x) stored(d)>
(a product of inner products is the inner product of the tensor products), so the gated clash is
still one exact inner product, with r times as many numbers per passage. It is high only for
passages on the same topic that clash: "similar overall and different in a few places".

Span slots (--span_k k > 0), for faithful explanations: each slot scores the tokens with the
encoder's CONTEXTUAL states, keeps only its top-k tokens, and computes its commitment and value
ONLY from those tokens' LOCAL features (token embedding plus immediate neighbours, a window of
--span_window). Nothing outside the selected tokens (and their window) can change a slot's value,
except by changing which tokens are selected: the selected tokens ARE the explanation.

Slot Attention (--slot_attention iters > 0; Locatello et al., 2020): each TOKEN distributes itself
over the slots (softmax over slots), so slots compete for words and must divide the passage, then
refine over a few iterations (GRU + MLP). Each slot starts from its own fixed, learned vector, so
slot c plays the same role in every passage: the precondition for types emerging across passages.

Single pass (--single_pass): the slot head reads the token states of the FROZEN cosine encoder
itself, so one encoder pass per text gives both the cosine embedding (its CLS token, as before)
and the slots; only the slot head (and the combination weights) are trained.

The retrieval score is  F(q, d) = cos(q, d) + alpha * K~(q, d),  where K~ = (w_K / w_cos) K uses
the weights learned in training (so alpha = 1 is the trained combination) and alpha is tuned on
the dev split exactly as SparseCL tunes its Hoyer weight. F is itself one inner product.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1e-9)


class CosEncoder(nn.Module):
    """Frozen embedder for the cosine term. pooling: cls (bge, gte: SparseCL's zero-shot setting)
    or mean."""

    def __init__(self, name, pooling=None, max_len=512):
        super().__init__()
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        low = name.lower()
        self.pooling = pooling or ("cls" if ("bge" in low or "gte" in low) else "mean")
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts, bs=64, device="cuda", amp=True):
        self.model.to(device)
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(texts[i:i + bs], max_length=self.max_len, padding=True, truncation=True,
                         return_tensors="pt").to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp and device.startswith("cuda")):
                h = self.model(**b).last_hidden_state
            e = h[:, 0] if self.pooling == "cls" else mean_pool(h, b["attention_mask"])
            out.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 1)


class SlotHead(nn.Module):
    """n learned queries cross-attend to the token states, then mix among themselves."""

    def __init__(self, d_in, n_slots, value_dim, d_model=256, n_heads=4, n_mix=2,
                 value_act="linear"):
        super().__init__()
        if value_act == "sphere" and value_dim < 2:
            raise ValueError("--value_act sphere needs --value_dim >= 2 (a 1-d unit value is "
                             "only +1 or -1 and has no gradient)")
        self.n, self.k, self.value_act = n_slots, value_dim, value_act
        self.proj = nn.Linear(d_in, d_model)
        self.queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, n_mix) if n_mix > 0 else nn.Identity()
        self.out = nn.Linear(d_model, 1 + value_dim)

    def forward(self, h, mask, return_attn=False):
        x = self.proj(h.float())
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        s, attn = self.cross(q, x, x, key_padding_mask=~mask.bool(), need_weights=return_attn)
        s = self.mix(self.norm(s + q))
        z = self.out(s)                                            # (B, n, 1 + k)
        a = torch.sigmoid(z[..., 0])
        # linear values cannot saturate: with tanh, training can switch K off by driving every
        # value into the flat part of tanh (all passages then share one value and K = 0)
        if self.value_act == "tanh":
            v = 2 * torch.tanh(z[..., 1:])
        elif self.value_act == "sphere":
            # unit-length values: every stored vector has nearly the same length, which approximate
            # inner-product search (reduced to nearest neighbours) needs; no saturation plateau
            v = F.normalize(z[..., 1:], dim=-1)
        else:
            v = z[..., 1:]
        if return_attn:
            return a, v, attn                   # attn: (B, n, T) token weights per slot
        return a, v


class SpanSlotHead(nn.Module):
    """Slots that read only their top-k tokens (selected with context, valued from local features).
    forward(h, mask, return_attn, emb, select_mask) -> a, v[, weights (B, n, T)]"""

    def __init__(self, d_ctx, d_emb, n_slots, value_dim, d_model=256, value_act="linear",
                 span_k=3, window=1, explore=0.5):
        super().__init__()
        self.n, self.k, self.value_act = n_slots, value_dim, value_act
        if value_act == "sphere" and value_dim < 2:
            raise ValueError("--value_act sphere needs --value_dim >= 2")
        self.span_k, self.explore = span_k, explore
        self.queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.keys = nn.Linear(d_ctx, d_model)
        self.local = nn.Conv1d(d_emb, d_model, kernel_size=2 * window + 1, padding=window)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, 1 + value_dim))

    def select(self, h, select_mask):
        """-> (B, n, T) weights: softmax over each slot's top-k tokens, exactly 0 elsewhere."""
        scores = torch.einsum("nd,btd->bnt", self.queries, self.keys(h.float())) / \
            math.sqrt(self.queries.size(1))
        scores = scores.masked_fill(~select_mask.unsqueeze(1), float("-inf"))
        pick = scores
        if self.training and self.explore > 0:                # try other tokens while training
            g = -torch.log(-torch.log(torch.rand_like(scores).clamp(1e-9, 1 - 1e-9)))
            pick = scores + self.explore * g
        k = max(1, min(self.span_k, int(select_mask.sum(1).min())))
        idx = pick.topk(k, dim=-1).indices
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True) & \
            select_mask.unsqueeze(1)
        w = torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1)
        return torch.nan_to_num(w, nan=0.0)

    def values(self, local, w):
        """slot commitments and values from the weighted local features of the selected tokens."""
        z = self.out(self.norm(torch.einsum("bnt,btd->bnd", w, local)))
        a = torch.sigmoid(z[..., 0])
        if self.value_act == "tanh":
            v = 2 * torch.tanh(z[..., 1:])
        elif self.value_act == "sphere":
            v = F.normalize(z[..., 1:], dim=-1)
        else:
            v = z[..., 1:]
        return a, v

    def local_features(self, emb, mask):
        x = (emb.float() * mask.unsqueeze(-1)).transpose(1, 2)            # (B, d_emb, T)
        return F.gelu(self.local(x)).transpose(1, 2) * mask.unsqueeze(-1)  # (B, T, d_model)

    def forward(self, h, mask, return_attn=False, emb=None, select_mask=None):
        w = self.select(h, select_mask if select_mask is not None else mask.bool())
        a, v = self.values(self.local_features(emb, mask), w)
        return (a, v, w) if return_attn else (a, v)


class SlotAttentionHead(nn.Module):
    """Competitive slots: softmax over SLOTS for each token, iterative refinement, fixed learned
    slot initialisations. forward(h, mask, return_attn, token_mask) -> a, v[, w (B, n, T)] where
    w is each slot's weighting of the tokens it won (normalised over tokens)."""

    def __init__(self, d_in, n_slots, value_dim, d_model=256, iters=3, value_act="linear"):
        super().__init__()
        self.n, self.k, self.iters, self.value_act = n_slots, value_dim, iters, value_act
        if value_act == "sphere" and value_dim < 2:
            raise ValueError("--value_act sphere needs --value_dim >= 2")
        self.slots_init = nn.Parameter(torch.randn(n_slots, d_model))
        self.norm_in = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_model)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model))
        self.out = nn.Linear(d_model, 1 + value_dim)
        self.last_mass = None

    def forward(self, h, mask, return_attn=False, token_mask=None):
        tm = (token_mask if token_mask is not None else mask.bool()).float()       # (B, T)
        x = self.proj(self.norm_in(h.float()))
        k, v = self.to_k(x), self.to_v(x)
        B, d = x.size(0), x.size(-1)
        slots = self.slots_init.unsqueeze(0).expand(B, -1, -1)
        for _ in range(self.iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))
            attn = torch.softmax(torch.einsum("btd,bnd->btn", k, q) / math.sqrt(d), dim=-1)
            attn = attn * tm.unsqueeze(-1)                     # padding / special tokens: no vote
            w = attn / (attn.sum(1, keepdim=True) + 1e-8)      # weighted mean over won tokens
            upd = torch.einsum("btn,btd->bnd", w, v)
            slots = self.gru(upd.reshape(-1, d), prev.reshape(-1, d)).reshape(B, self.n, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        self.last_mass = attn.sum(1) / tm.sum(1, keepdim=True).clamp_min(1)        # (B, n)
        z = self.out(slots)
        a = torch.sigmoid(z[..., 0])
        if self.value_act == "tanh":
            vv = 2 * torch.tanh(z[..., 1:])
        elif self.value_act == "sphere":
            vv = F.normalize(z[..., 1:], dim=-1)
        else:
            vv = z[..., 1:]
        return (a, vv, w.transpose(1, 2)) if return_attn else (a, vv)


class SlotEncoder(nn.Module):
    def __init__(self, backbone, n_slots=16, value_dim=8, d_model=256, n_heads=4, n_mix=2,
                 max_len=512, value_act="linear", gate_dim=16, cos_dim=None,
                 init_wk=1.0, init_wcos=50.0, frozen_backbone=False, cos_pooling="cls",
                 span_k=0, span_window=1, slot_attention=0):
        super().__init__()
        self.config = dict(backbone=backbone, n_slots=n_slots, value_dim=value_dim,
                           d_model=d_model, n_heads=n_heads, n_mix=n_mix, max_len=max_len,
                           value_act=value_act, gate_dim=gate_dim, cos_dim=cos_dim,
                           init_wk=init_wk, init_wcos=init_wcos,
                           frozen_backbone=frozen_backbone, cos_pooling=cos_pooling,
                           span_k=span_k, span_window=span_window,
                           slot_attention=slot_attention)
        if gate_dim and not cos_dim:
            raise ValueError("gate_dim needs cos_dim (the frozen cosine embedding size)")
        self.gate = nn.Linear(cos_dim, gate_dim) if gate_dim else None
        self.tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(backbone, trust_remote_code=True)
        self.span = span_k > 0
        self.slot_attn = slot_attention > 0
        if self.span and self.slot_attn:
            raise ValueError("choose --span_k or --slot_attention, not both")
        if self.span or self.slot_attn:
            self.register_buffer("special_ids", torch.tensor(sorted(set(self.tok.all_special_ids))),
                                 persistent=False)
        if self.slot_attn:
            self.head = SlotAttentionHead(self.encoder.config.hidden_size, n_slots, value_dim,
                                          d_model, slot_attention, value_act)
        elif self.span:
            d = self.encoder.config.hidden_size
            self.head = SpanSlotHead(d, self.encoder.get_input_embeddings().embedding_dim,
                                     n_slots, value_dim, d_model, value_act, span_k, span_window)
        else:
            self.head = SlotHead(self.encoder.config.hidden_size, n_slots, value_dim, d_model,
                                 n_heads, n_mix, value_act)
        # learned combination weights: logit = w_K * K + w_cos * cos (both positive). w_cos starts
        # at 50 (SparseCL's temperature 0.02); w_K starts small, because at initialisation K is
        # noise of order 1 per pair: w_K * K would otherwise swamp the cosine signal, and training
        # then learns to switch K off instead of making it informative.
        self.log_wk = nn.Parameter(torch.tensor(float(init_wk)).log())
        self.log_wc = nn.Parameter(torch.tensor(float(init_wcos)).log())
        self.n, self.k = n_slots, value_dim
        self.frozen_backbone = frozen_backbone
        self.cos_pooling = cos_pooling
        if frozen_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:
            self.encoder.eval()                 # a frozen encoder never uses dropout
        return self

    def tokenize(self, texts, device):
        return self.tok(texts, max_length=self.config["max_len"], padding=True, truncation=True,
                        return_tensors="pt").to(device)

    def hidden(self, batch):
        if self.frozen_backbone:
            with torch.no_grad():
                return self.encoder(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"]).last_hidden_state
        return self.encoder(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"]).last_hidden_state

    def run_head(self, h, batch, return_attn=False, emb=None):
        """emb: optional input embeddings to use for span slots' local features (the probe passes
        the tensor it differentiates, so attributions see both the selection and the values)."""
        mask = batch["attention_mask"]
        if self.slot_attn:
            ids = batch["input_ids"]
            tm = mask.bool() & ~torch.isin(ids, self.special_ids.to(ids.device))
            return self.head(h, mask, return_attn, token_mask=tm)
        if not self.span:
            return self.head(h, mask, return_attn)
        ids = batch["input_ids"]
        if emb is None:
            emb = self.encoder.get_input_embeddings()(ids)
        select = mask.bool() & ~torch.isin(ids, self.special_ids.to(ids.device))
        return self.head(h, mask, return_attn, emb=emb, select_mask=select)

    def forward(self, batch, return_attn=False):
        return self.run_head(self.hidden(batch), batch, return_attn)

    @torch.no_grad()
    def encode_both(self, texts, bs=64, device="cuda", amp=True, side="stored"):
        """single-pass models only: ONE encoder pass per text -> (cosine embeddings, slot vectors)."""
        if not self.frozen_backbone:
            raise ValueError("encode_both needs a single-pass model (--single_pass)")
        self.eval().to(device)
        us, vecs = [], []
        for i in range(0, len(texts), bs):
            b = self.tokenize(texts[i:i + bs], device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp and device.startswith("cuda")):
                h = self.hidden(b)
                a, v = self.run_head(h, b)
            e = h[:, 0] if self.cos_pooling == "cls" else mean_pool(h, b["attention_mask"])
            us.append(F.normalize(e.float(), dim=-1).cpu())
            a, v = a.float(), v.float()
            vecs.append((self.stored_vec(a, v) if side == "stored" else self.query_vec(a, v)).cpu())
        return torch.cat(us), torch.cat(vecs)

    # ---- the clash term, directly and as an inner product ----
    def clash(self, aq, vq, ad, vd):
        """(Bq, n), (Bq, n, k), (Bd, n), (Bd, n, k) -> (Bq, Bd), the direct formula."""
        diff = (vd.unsqueeze(0) - vq.unsqueeze(1)).pow(2).sum(-1)  # (Bq, Bd, n)
        return (aq.unsqueeze(1) * ad.unsqueeze(0) * diff).sum(-1) / self.n

    def clash_per_slot(self, aq, vq, ad, vd):
        """aligned pairs (B, n), (B, n, k) x2 -> (B, n): each slot's share a^q a^d ||v^d - v^q||^2
        (sums to n * K over slots)."""
        return aq * ad * (vd - vq).pow(2).sum(-1)

    def stored_vec(self, a, v):
        return torch.cat([a * v.pow(2).sum(-1), a, (a.unsqueeze(-1) * v).flatten(1)], dim=1)

    def query_vec(self, a, v):
        return torch.cat([a, a * v.pow(2).sum(-1), (-2 * a.unsqueeze(-1) * v).flatten(1)],
                         dim=1) / self.n

    def topic(self, u):
        """frozen cosine embeddings (N, cos_dim) -> unit topic vectors (N, gate_dim)."""
        return F.normalize(self.gate(u.float()), dim=-1)

    def gated_clash(self, aq, vq, ad, vd, uq, ud):
        K = self.clash(aq, vq, ad, vd)
        return K if self.gate is None else K * (self.topic(uq) @ self.topic(ud).T)

    def gate_vectors(self, vec, u):
        """stored/query vectors (N, m) + cosine embeddings (N, cos_dim) -> (N, gate_dim * m)."""
        if self.gate is None:
            return vec
        dev = self.gate.weight.device                # wherever the model lives (GPU after encode)
        out = []
        for i in range(0, vec.size(0), 4096):        # chunked: the result is gate_dim times wider
            t = self.topic(u[i:i + 4096].to(dev))
            v = vec[i:i + 4096].to(dev)
            out.append((t.unsqueeze(2) * v.unsqueeze(1)).flatten(1).to(vec.device))
        return torch.cat(out) if out else vec.new_zeros(0, vec.size(1) * self.gate.out_features)

    def k_scale(self):
        """K~ = (w_K / w_cos) K: alpha = 1 then reproduces the trained combination."""
        return float((self.log_wk - self.log_wc).exp())

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=1)

    @classmethod
    def load(cls, path, map_location="cpu"):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        m = cls(**cfg)
        m.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=map_location))
        return m

    @torch.no_grad()
    def encode(self, texts, bs=64, device="cuda", amp=True, side="stored"):
        """-> (N, 2n + n k) stored or query vectors."""
        self.eval().to(device)
        out = []
        for i in range(0, len(texts), bs):
            b = self.tokenize(texts[i:i + bs], device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=amp and device.startswith("cuda")):
                a, v = self(b)
            a, v = a.float(), v.float()
            out.append((self.stored_vec(a, v) if side == "stored" else self.query_vec(a, v)).cpu())
        return torch.cat(out)
