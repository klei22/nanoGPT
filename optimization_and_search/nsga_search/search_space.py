# hetero_space.py

import random, math
from typing import Any, Dict, List, TypedDict

class Individual(dict):
    """Runtime Individual object that behaves like a dict but has helpers."""
    def __init__(self, globals: Dict[str, Any] = None, layers: List[Dict[str, Any]] = None):
        super().__init__()
        self["globals"] = globals or {}
        self["layers"] = layers or []

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Individual":
        return Individual(dict(d.get("globals", {})), list(d.get("layers", [])))

    def estimate_params(self) -> int:
        g = self["globals"]; d = g["d_model"]; seq_len = g["block_size"]

        # Embedding table parameters (same as GPT-2)
        vocab_size = 50257  # GPT-2 vocabulary size
        embedding_params = vocab_size * d  # token embeddings: vocab_size x d_model
        # embedding_params += seq_len * d    # positional embeddings: block_size x d_model
        
        total = embedding_params
        mask = self["globals"].get("layer_mask", [True]*len(self["layers"]))
        for i,active in enumerate(mask):
            if not active: continue
            li = self["layers"][i]
            h = max(1, li["n_heads"])
            r = li["mlp_ratio"]
            # very rough: per-layer params ~ 2*d^2 (proj) + r*d^2 (MLP) + heads overhead
            total += (2.0 + r)*d*d + 0.05*h*d*d
        return int(total)

    def print_individual(self, include_inactive: bool = False, include_params: bool = True, max_layers: int = None) -> None:
        """Return a human-readable, layer-aware summary of this Individual.

        - include_inactive: when True, also lists layers masked off (marked as [inactive])
        - include_params: when True, appends an estimated parameter count if available
        - max_layers: optional cap on how many layers to print (useful for very deep nets)
        """
        g = self.get("globals", {})
        layers: List[Dict[str, Any]] = self.get("layers", [])
        mask: List[bool] = g.get("layer_mask", [True] * len(layers))
        active_count = sum(1 for i in range(min(len(mask), len(layers))) if mask[i])
        header = (
            f"Individual: d_model={g.get('d_model')}, block_size={g.get('block_size')}, "
            f"quant_bits={g.get('quant_bits')}, active_layers={active_count}/{len(layers)}"
        )
        lines = [header]
        if include_params:
            try:
                params = self.estimate_params()
                lines.append(f"  ~params={params/1e6:.1f}M")
            except Exception:
                pass
        lines.append("  Layers:")
        limit = max_layers if isinstance(max_layers, int) and max_layers >= 0 else len(layers)
        for idx, li in enumerate(layers[:limit]):
            active = mask[idx] if idx < len(mask) else True
            if not include_inactive and not active:
                continue
            lines.append(
                f"    - L{idx:02d}: n_heads={li.get('n_heads')}, mlp_ratio={li.get('mlp_ratio')}, "
                f"attn_type={li.get('attn_type')} {'[inactive]' if not active else ''}"
            )
        if limit < len(layers):
            lines.append(f"    ... (+{len(layers) - limit} more layers)")

        print("\n".join(lines))
        return

class HeteroSearchSpace:
    def __init__(self, L_max=24):
        self.L_max = L_max

        # Globals
        self.globals = {
            "d_model":      {"type":"int","low":256,"high":2048,"step":256},
            "block_size":      {"type":"int","low":512,"high":512,"step":128},
            "quant_bits":   {"type":"int","low":8,"high":8},
            # replaced active_L with an explicit layer usage mask of length L_max
            #"layer_mask" added later
        }

        # Per-layer fields (heterogeneous)
        self.layer_spec = {
            "n_heads":    {"type":"int","low":1,"high":32,"step":1},   # will be clamped to divisor of d_model
            "mlp_ratio":  {"type":"int","low":1,"high":8,"step":1},
            "attn_type":  {"type":"cat","choices":["mha"]},
        }

    # ---------- utils ----------
    def _sample_global(self):
        g = {}
        for k,s in self.globals.items():
            if s["type"]=="int":
                step=s.get("step",1)
                g[k]=random.randrange(s["low"], s["high"]+1, step)
            elif s["type"]=="float":
                g[k]=random.uniform(s["low"], s["high"])
            elif s["type"]=="cat":
                g[k]=random.choice(s["choices"])
        return g

    def _sample_layer(self):
        l = {}
        for k,s in self.layer_spec.items():
            # ensure head is to the power of 2 for efficiency
            if k == "n_heads":
                choices = [h for h in range(s["low"], s["high"]+1) if (h & (h - 1)) == 0]
                l[k] = random.choice(choices) if choices else s["low"]
                continue

            if s["type"]=="int":
                step=s.get("step",1)
                l[k]=random.randrange(s["low"], s["high"]+1, step)
            elif s["type"]=="float":
                l[k]=random.uniform(s["low"], s["high"])
            elif s["type"]=="cat":
                l[k]=random.choice(s["choices"])
        return l

    # ---------- public API ----------
    def sample(self) -> Individual:
        g = self._sample_global()
        layers = [self._sample_layer() for _ in range(self.L_max)]
        x: Individual = Individual(g, layers)
        # if globals does not yet have layer_mask (e.g., older serialized), create one
        if "layer_mask" not in x["globals"]:
            min_active = 4
            active_count = random.randint(min_active, self.L_max)
            idxs = set(random.sample(range(self.L_max), active_count))
            x["globals"]["layer_mask"] = [i in idxs for i in range(self.L_max)]
        return self.repair(x)

    def repair(self, x: Dict[str, Any]) -> Individual:
        # assume mask always provided; if missing, default to all active
        if "globals" not in x:
            x["globals"] = {}
        if "layer_mask" not in x["globals"]:
            x["globals"]["layer_mask"] = [True]*self.L_max
        mask = list(x["globals"]["layer_mask"])[:self.L_max]
        if len(mask) < self.L_max:
            mask.extend([False]*(self.L_max-len(mask)))
        x["globals"]["layer_mask"] = mask

        y: Dict[str, Any] = {"globals": dict(x["globals"]), "layers": [dict(li) for li in x["layers"]] }
        # clamp globals
        for k,s in self.globals.items():
            if s["type"]=="int":
                step=s.get("step",1)
                lo,hi=s["low"],s["high"]
                y["globals"][k]=max(lo, min(hi, round(y["globals"][k]/step)*step))
            elif s["type"]=="float":
                lo,hi=s["low"],s["high"]
                y["globals"][k]=float(max(lo, min(hi, y["globals"][k])))
            elif s["type"]=="cat":
                if y["globals"][k] not in s["choices"]:
                    y["globals"][k]=s["choices"][0]

        d_model = y["globals"]["d_model"]
        # clamp per-layer + divisibility
        for li in y["layers"]:
            for k,s in self.layer_spec.items():
                if s["type"]=="int":
                    step=s.get("step",1)
                    lo,hi=s["low"],s["high"]
                    li[k]=max(lo, min(hi, round(li[k]/step)*step))
                elif s["type"]=="float":
                    lo,hi=s["low"],s["high"]
                    li[k]=float(max(lo, min(hi, li[k])))
                elif s["type"]=="cat":
                    if li[k] not in s["choices"]:
                        li[k]=s["choices"][0]
            # enforce d_model % n_heads == 0
            if d_model % li["n_heads"] != 0:
                # snap to nearest divisor in [low, high]
                candidates = [h for h in range(s["low"], s["high"]+1) if (h & (h - 1)) == 0 and d_model % h == 0]
                if candidates:
                    li["n_heads"] = min(candidates, key=lambda h: abs(h - li["n_heads"]))
        # ensure at least one active layer; if mask empty, activate first 4 or available
        if not any(y["globals"]["layer_mask"]):
            for i in range(min(4, self.L_max)):
                y["globals"]["layer_mask"][i] = True
        return Individual.from_dict(y)

    # ----- variation: layer-aware -----
    def crossover(self, a: Dict[str,Any], b: Dict[str,Any], crossover_rate: float = 0.5) -> (Dict[str,Any], Dict[str,Any]):
        A = {"globals": dict(a["globals"]), "layers":[dict(li) for li in a["layers"]]}
        B = {"globals": dict(b["globals"]), "layers":[dict(li) for li in b["layers"]]}

        # uniform crossover on globals
        for k in self.globals:
            if random.random() < crossover_rate:
                A["globals"][k], B["globals"][k] = B["globals"][k], A["globals"][k]

        # layer usage mask crossover (treat mask as gene string)
        mask_a = list(a["globals"].get("layer_mask", [True]*self.L_max))
        mask_b = list(b["globals"].get("layer_mask", [True]*self.L_max))
        # single point crossover on mask
        cut = random.randint(1, self.L_max-1) if self.L_max>1 else 0
        new_mask_a = mask_a[:cut] + mask_b[cut:]
        new_mask_b = mask_b[:cut] + mask_a[cut:]
        A["globals"]["layer_mask"] = new_mask_a
        B["globals"]["layer_mask"] = new_mask_b

        # segment crossover on layers
        L = self.L_max
        a_idx, b_idx = sorted(random.sample(range(L), 2))
        for i in range(a_idx, b_idx+1):
            A["layers"][i], B["layers"][i] = B["layers"][i], A["layers"][i]

        return self.repair(A), self.repair(B)

    def mutate(self, x: Dict[str,Any],
               p_glob_int=0.1, p_glob_float=0.1,
               p_layer_int=0.08, p_layer_float=0.08, p_layer_cat=0.05,
               p_swap_layers=0.05) -> Dict[str,Any]:
        y = {"globals": dict(x["globals"]), "layers":[dict(li) for li in x["layers"]]}

        # mutate globals
        for k,s in self.globals.items():
            if s["type"]=="int" and random.random()<p_glob_int:
                step=s.get("step",1); lo,hi=s["low"],s["high"]
                span=((hi-lo)//step)+1
                delta=random.randint(-max(1, span//8), max(1, span//8))*step
                y["globals"][k]=max(lo,min(hi, y["globals"][k]+delta))
            elif s["type"]=="float" and random.random()<p_glob_float:
                lo,hi=s["low"],s["high"]
                sigma=(hi-lo)*0.05
                y["globals"][k]=max(lo,min(hi, y["globals"][k]+random.gauss(0,sigma)))

        # mutate layers
        for li in y["layers"]:
            # int
            if random.random()<p_layer_int:
                s=self.layer_spec["n_heads"]; lo,hi=s["low"],s["high"]; step=s.get("step",1)
                span=((hi-lo)//step)+1
                delta=random.randint(-2,2)*step
                li["n_heads"]=max(lo,min(hi, li["n_heads"]+delta))
            # float
            if random.random()<p_layer_float:
                s=self.layer_spec["mlp_ratio"]; lo,hi=s["low"],s["high"]
                sigma=(hi-lo)*0.05
                li["mlp_ratio"]=max(lo,min(hi, li["mlp_ratio"]+random.gauss(0,sigma)))
            # categorical
            if random.random()<p_layer_cat:
                choices=self.layer_spec["attn_type"]["choices"]
                cur=li["attn_type"]
                li["attn_type"]=random.choice([c for c in choices if c!=cur] or choices)

        # occasional layer swap (explores schedule if meaningful)
        if random.random()<p_swap_layers and self.L_max>=2:
            i,j = random.sample(range(self.L_max), 2)
            y["layers"][i], y["layers"][j] = y["layers"][j], y["layers"][i]

        # mutate layer usage mask: flip a few bits
        mask = list(x["globals"].get("layer_mask", [True]*self.L_max))
        flips = max(1, self.L_max//10)
        for _ in range(random.randint(1, flips)):
            idx = random.randrange(self.L_max)
            mask[idx] = not mask[idx]
        # ensure still at least four active
        min_layers = 4  
        if sum(mask) < min_layers:
            for _ in range(min_layers - sum(mask)):
                mask[random.randrange(self.L_max)] = True
        y["globals"]["layer_mask"] = mask
        return self.repair(y)

    def estimate_params_hetero(x):
        g = x["globals"]; d = g["d_model"]; seq_len = g["block_size"]
        
        # Embedding table parameters (same as GPT-2)
        vocab_size = 50257  # GPT-2 vocabulary size
        embedding_params = vocab_size * d  # token embeddings: vocab_size x d_model
        # embedding_params += seq_len * d    # positional embeddings: block_size x d_model
        
        total = embedding_params
        mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
        for i,active in enumerate(mask):
            if not active: continue
            li = x["layers"][i]
            h = max(1, li["n_heads"])
            r = li["mlp_ratio"]
            # very rough: per-layer params ~ 2*d^2 (proj) + r*d^2 (MLP) + heads overhead
            total += (2.0 + r)*d*d + 0.05*h*d*d
        return int(total)

    def estimate_flops_hetero(x):
        g = x["globals"]; d = g["d_model"]; seq = g["block_size"]
        cost = 0.0
        mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
        for i,active in enumerate(mask):
            if not active: continue
            li = x["layers"][i]
            attn = li["attn_type"]
            attn_cost = {"scaled_dot": d*seq,
                        "gqa": d*seq/2,
                        "mha": d*math.log2(max(2, seq)),
                        "flash": d*seq*0.7}.get(attn, d*seq)
            cost += (2*d*d + attn_cost) + li["mlp_ratio"]*d*d
        return cost

