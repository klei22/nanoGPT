# hetero_space.py

import random, math
from typing import Any, Dict, List, Tuple, TypedDict

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
        # g = self["globals"]; d = g["d_model"]; seq_len = g["block_size"]

        # # Embedding table parameters (same as GPT-2)
        # vocab_size = 50257  # GPT-2 vocabulary size
        # embedding_params = vocab_size * d  # token embeddings: vocab_size x d_model
        # # embedding_params += seq_len * d    # positional embeddings: block_size x d_model
        
        # total = embedding_params
        # mask = self["globals"].get("layer_mask", [True]*len(self["layers"]))
        # for i,active in enumerate(mask):
        #     if not active: continue
        #     li = self["layers"][i]
        #     h = max(1, li["n_heads"])
        #     r = li["mlp_ratio"]
        #     # very rough: per-layer params ~ 2*d^2 (proj) + r*d^2 (MLP) + heads overhead
        #     total += (2.0 + r)*d*d + 0.05*h*d*d
        # return int(total)
        x = self
        g = x["globals"]
        d = g.get("n_embd", g.get("d_model", 768))
        total = 0.0

        # Embedding table (approx)
        vocab_size = 50257
        total += vocab_size * d

        mask = g.get("layer_mask", [True] * len(x["layers"]))
        indices = [i for i, active in enumerate(mask) if active and i < len(x["layers"])]
        for i in indices:
            li = x["layers"][i]
            h = int(li.get("n_head", 8))
            m = int(li.get("mlp_size", 4 * d))
            qk = int(li.get("n_qk_head_dim", d // max(1, h)))
            v = int(li.get("n_v_head_dim", d // max(1, h)))
            n_cproj = int(li.get("n_cproj", 1))
            n_kv_group = int(li.get("n_kv_group", h))
            use_concat_heads = bool(li.get("use_concat_heads", g.get("use_concat_heads", False)))
            attn_variant = li.get("attention_variant", g.get("attention_variant", "mha"))
            
            attn_cost = 0
            
            if attn_variant == "infinite":
                # Q, K, V projection weights
                q_params = d * (h * qk)
                k_params = d * (n_kv_group * qk)
                v_params = d * (n_kv_group * v)

                # Output projection params
                if use_concat_heads:
                    # concatenate all heads then project back to d
                    cproj_params = (h * v) * d
                else:
                    # sum heads first; project v_head_dim back to d, possibly with multiple small projections
                    cproj_params = n_cproj * (v * d)

                attn_cost = q_params + k_params + v_params + cproj_params
            elif attn_variant == "mha":
                # QKV projection weights
                qkv_params = d * (h * (qk + qk + v))
                # Output projection params
                if use_concat_heads:
                    out_proj_params = (h * v) * d
                else:
                    out_proj_params = n_cproj * (v * d)
                attn_cost = qkv_params + out_proj_params
            elif attn_variant == "identity":
                attn_cost = 0
            else:
                raise ValueError(f"Unknown attention_variant: {attn_variant}")

            # MLP params (two linear layers d->m and m->d)
            mlp_params = 2 * d * m

            total += attn_cost + mlp_params

        return int(total)
    
    def estimate_flops(self, seq_len: int = 512) -> int:
        x = self
        g = x["globals"]
        d = g.get("n_embd", g.get("d_model", 768))
        seq = int(g.get("block_size", 512))
        cost = 0.0

        mask = g.get("layer_mask", [True] * len(x["layers"]))
        indices = [i for i, active in enumerate(mask) if active and i < len(x["layers"])]
        for i in indices:
            li = x["layers"][i]
            h = int(li.get("n_head", 8))
            m = int(li.get("mlp_size", 4 * d))
            qk = int(li.get("n_qk_head_dim", d // max(1, h)))
            v = int(li.get("n_v_head_dim", d // max(1, h)))
            n_cproj = int(li.get("n_cproj", 1))
            n_kv_group = int(li.get("n_kv_group", h))
            use_concat_heads = bool(li.get("use_concat_heads", g.get("use_concat_heads", False)))

            attn_cost = 0.0
            attn_variant = li.get("attention_variant", g.get("attention_variant", "mha"))
            
            if attn_variant == "infinite":
                # Q, K, V projections
                q_proj = 2.0 * seq * d * (h * qk)
                k_proj = 2.0 * seq * d * (n_kv_group * qk)
                v_proj = 2.0 * seq * d * (n_kv_group * v)

                # Attention core (QK^T, softmax, weighted sum)
                attn_core = 2.0 * h * seq * (seq / n_kv_group) * qk + 2.0 * h * seq * (seq / n_kv_group)

                # Output projection
                if use_concat_heads:
                    outp = 2.0 * seq * (h * v) * d
                else:
                    outp = n_cproj * (2.0 * seq * v * d)

                attn_cost = q_proj + k_proj + v_proj + attn_core + outp
                
            elif attn_variant == "mha":
                # QKV projections
                qkv_proj = 2.0 * seq * d * (h * (qk + qk + v))
                # Attention core
                attn_core = 2.0 * h * seq * seq * qk + 2.0 * h * seq * seq
                # Output projection
                if use_concat_heads:
                    outp = 2.0 * seq * (h * v) * d
                else:
                    outp = n_cproj * (2.0 * seq * v * d)
                attn_cost = qkv_proj + attn_core + outp
            elif attn_variant == "identity":
                attn_cost = 0.0
            else:
                raise ValueError(f"Unknown attention_variant: {attn_variant}")
            mlp = 4.0 * seq * d * m

            cost += attn_cost + mlp
        return cost
    
    def estimate_mem_access(self, seq_len: int = 512) -> int:
        x = self
        g = x["globals"]
        d = g.get("n_embd", g.get("d_model", 768))
        seq = int(g.get("block_size", 512))
        cost = 0.0

        mask = g.get("layer_mask", [True] * len(x["layers"]))
        indices = [i for i, active in enumerate(mask) if active and i < len(x["layers"])]
        for i in indices:
            li = x["layers"][i]
            h = int(li.get("n_head", 8))
            m = int(li.get("mlp_size", 4 * d))
            qk = int(li.get("n_qk_head_dim", d // max(1, h)))
            v = int(li.get("n_v_head_dim", d // max(1, h)))
            n_cproj = int(li.get("n_cproj", 1))
            n_kv_group = int(li.get("n_kv_group", h))
            use_concat_heads = bool(li.get("use_concat_heads", g.get("use_concat_heads", False)))

            attn_cost = 0.0
            attn_variant = li.get("attention_variant", g.get("attention_variant", "mha"))
            
            # consider reads and writes to WMEM and KV CACHE
            if attn_variant == "infinite":
                # Q, K, V projections
                q_proj = 2.0 * seq * d * (h * qk)
                k_proj = 2.0 * seq * d * (n_kv_group * qk)
                v_proj = 2.0 * seq * d * (n_kv_group * v)

                # Attention core (QK^T, softmax, weighted sum)
                attn_core = 2.0 * h * seq * (seq / n_kv_group) * qk + 2.0 * h * seq * (seq / n_kv_group)

                # Output projection
                if use_concat_heads:
                    outp = 2.0 * seq * (h * v) * d
                else:
                    outp = n_cproj * (2.0 * seq * v * d)

                attn_cost = q_proj + k_proj + v_proj + attn_core + outp

            elif attn_variant == "mha":
                # QKV projections
                qkv_proj = 2.0 * seq * d * (h * (qk + qk + v))
                # Attention core
                attn_core = 2.0 * h * seq * seq * qk + 2.0 * h * seq * seq
                # Output projection
                if use_concat_heads:
                    outp = 2.0 * seq * (h * v) * d
                else:
                    outp = n_cproj * (2.0 * seq * v * d)
                attn_cost = qkv_proj + attn_core + outp
            elif attn_variant == "identity":
                attn_cost = 0.0
            else:
                raise ValueError(f"Unknown attention_variant: {attn_variant}")
            mlp = 4.0 * seq * d * m

            cost += attn_cost + mlp
        return cost
        
    

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
        
        print(f"Globals: {g}")
        print(f"Total layers: {len(layers)}; Active layers: {active_count}")
        if include_params:
            try:
                params = self.estimate_params()
                print(f"Estimated params: {params/1e6:.2f}M")
            except Exception as e:
                print(f"Estimated params: Error ({e})")
        for i, layer in enumerate(layers):
            if max_layers is not None and i >= max_layers:
                print(f"... (only showing first {max_layers} layers)")
                break
            active = mask[i] if i < len(mask) else True
            # skip inactive layers unless requested
            if not active and not include_inactive:
                continue
            status = "" if active else " [inactive]"
            # include per-layer params always in one line
            params = layer.get("params", {})
            param_str = f" Params: {params}" if params else ""
            # format layer dict as comma-separated key=value pairs for readability
            kv_pairs = []
            for k, v in layer.items():
                # skip the params field since it's printed separately
                if k == "params":
                    continue
                kv_pairs.append(f"{k}={v}")
            layer_str = ", ".join(kv_pairs)
            print(f"  - Layer {i}: {layer_str}{status}{param_str}")
            
        return

class HeteroSearchSpace:
    def __init__(self, L_max=24, L_min=1):
        self.L_max = L_max
        self.L_min = L_min  # minimum active layers

        self.no_repair = False  # if True, sample() does not call repair()

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

    @classmethod
    def from_dicts(
        cls,
        globals_spec: Dict[str, Any],
        layer_spec: Dict[str, Any],
        L_max: int = 24,
        L_min: int = 1, 
        no_repair: bool = False
    ) -> "HeteroSearchSpace":
        """Alternate constructor: build a search space from explicit spec dicts.

        Parameters
        - globals_spec: dict mapping global field name -> spec
            Spec format examples:
              {"type": "int", "low": 256, "high": 2048, "step": 256}
              {"type": "float", "low": 0.0, "high": 1.0}
              {"type": "cat", "choices": ["mha", "flash"]}
        - layer_spec: dict mapping per-layer field name -> spec (same format as above)
        - L_max: maximum number of layers

        Returns a configured HeteroSearchSpace instance.
        """
        inst = cls(L_max=L_max, L_min=L_min)
        inst.globals = cls._normalize_spec_dict(globals_spec)
        inst.no_repair = no_repair
        if layer_spec is None:
            inst.layer_spec = {}
        else:
            inst.layer_spec = cls._normalize_spec_dict(layer_spec)
        return inst

    @staticmethod
    def _normalize_spec_dict(spec_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize a spec dict.

        Ensures required keys exist and fills reasonable defaults (e.g., step=1 for ints).
        Raises ValueError on invalid specifications.
        """
        out: Dict[str, Any] = {}
        for k, raw in spec_dict.items():
            if not isinstance(raw, dict):
                raise ValueError(f"Spec for '{k}' must be a dict, got {type(raw)}")
            s = dict(raw)  # shallow copy
            s_type = s.get("type")
            if s_type not in {"int", "float", "cat"}:
                raise ValueError(
                    f"Spec for '{k}' must include a valid 'type' in {{'int','float','cat'}}, got {s_type}"
                )
            if s_type == "int":
                for req in ("low", "high"):
                    if req not in s:
                        raise ValueError(f"Int spec for '{k}' missing '{req}'")
                s.setdefault("step", 1)
                if not isinstance(s["step"], int) or s["step"] <= 0:
                    raise ValueError(f"Int spec for '{k}' has invalid step: {s['step']}")
            elif s_type == "float":
                for req in ("low", "high"):
                    if req not in s:
                        raise ValueError(f"Float spec for '{k}' missing '{req}'")
            elif s_type == "cat":
                if "choices" not in s or not s["choices"]:
                    raise ValueError(f"Cat spec for '{k}' requires non-empty 'choices'")
            out[k] = s
        return out

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
    
    def print_search_space(self) -> None:
        print("Global parameters:")
        for k, s in self.globals.items():
            print(f"  - {k}: {s}")
        print(f"Per-layer parameters (L_max={self.L_max}):")
        for k, s in self.layer_spec.items():
            print(f"  - {k}: {s}")
        print(f"Minimum active layers (L_min): {self.L_min}")
        print(f"No repair mode: {self.no_repair}")
        return

    # ---------- public API ----------
    def sample(self) -> Individual:
        g = self._sample_global()
        layers = [self._sample_layer() for _ in range(self.L_max)]
        x: Individual = Individual(g, layers)
        # if globals does not yet have layer_mask (e.g., older serialized), create one
        if "layer_mask" not in x["globals"]:
            active_count = random.randint(self.L_min, self.L_max)
            idxs = set(random.sample(range(self.L_max), active_count))
            x["globals"]["layer_mask"] = [i in idxs for i in range(self.L_max)]
        
        return self.repair(x)

    def repair(self, x: Dict[str, Any]) -> Individual:
        # assume mask always provided; if missing, default to all active
        if self.no_repair:
            return Individual.from_dict(x)
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

        # n_embd = y["globals"]["n_embd"]
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
            # enforce n_embd % n_head == 0
            attn_type = li.get("attn_variant", x["globals"].get("attn_variant", "mha"))
            if attn_type == "mha":
                n_head = li.get("n_head", 8)
                n_embd = y["globals"].get("n_embd", 768)
                if n_embd % n_head != 0:
                    # snap n_head to nearest divisor of n_embd
                    divisors = [h for h in range(1, min(n_head, n_embd)+1) if n_embd % h == 0]
                    if divisors:
                        closest = min(divisors, key=lambda h: abs(h - n_head))
                        li["n_head"] = closest
                    else:
                        li["n_head"] = 1  # fallback
                        
            # if n_kv_group is defined, ensure it divides n_head
            if "n_kv_group" in li:
                n_kv_group = li.get("n_kv_group", 1)
                n_head = li.get("n_head", 8)
                if n_head % n_kv_group != 0:
                    divisors = [g for g in range(1, min(n_kv_group, n_head)+1) if n_head % g == 0]
                    if divisors:
                        closest = min(divisors, key=lambda g: abs(g - n_kv_group))
                        li["n_kv_group"] = closest
                    else:
                        li["n_kv_group"] = 1  # fallback
                
            
        # ensure at least one active layer; if mask empty, activate first 4 or available
        if not any(y["globals"]["layer_mask"]):
            for i in range(min(4, self.L_max)):
                y["globals"]["layer_mask"][i] = True
        return Individual.from_dict(y)

    # ----- variation: layer-aware -----
    def crossover(self, a: Dict[str,Any], b: Dict[str,Any], crossover_rate: float = 0.5) -> Tuple[Dict[str,Any], Dict[str,Any]]:
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
                span_steps=max(1, (hi-lo)//step)
                sigma_steps=max(1.0, span_steps/8.0)
                delta_steps=int(round(random.gauss(0.0, sigma_steps)))
                new_val = y["globals"][k] + delta_steps*step
                # snap back to step grid and clamp
                new_val = int(round(new_val/step)*step)
                y["globals"][k]=max(lo,min(hi, new_val))
            elif s["type"]=="float" and random.random()<p_glob_float:
                lo,hi=s["low"],s["high"]
                sigma=(hi-lo)*0.05
                y["globals"][k]=max(lo,min(hi, y["globals"][k]+random.gauss(0,sigma)))

        # mutate layers
        for li in y["layers"]:
            # make it generic
            for k,s in self.layer_spec.items():
                if s["type"]=="int" and random.random()<p_layer_int:
                    step=s.get("step",1); lo,hi=s["low"],s["high"]
                    span_steps=max(1, (hi-lo)//step)
                    sigma_steps=max(1.0, span_steps/8.0)
                    delta_steps=int(round(random.gauss(0.0, sigma_steps)))
                    new_val = li[k] + delta_steps*step
                    new_val = int(round(new_val/step)*step)
                    li[k]=max(lo,min(hi, new_val))
                elif s["type"]=="float" and random.random()<p_layer_float:
                    lo,hi=s["low"],s["high"]
                    sigma=(hi-lo)*0.05
                    li[k]=max(lo,min(hi, li[k]+random.gauss(0,sigma)))
                elif s["type"]=="cat" and random.random()<p_layer_cat:
                    choices=s["choices"]
                    cur=li[k]
                    li[k]=random.choice([c for c in choices if c!=cur] or choices)
            # # int
            # if random.random()<p_layer_int:
            #     s=self.layer_spec["n_heads"]; lo,hi=s["low"],s["high"]; step=s.get("step",1)
            #     span=((hi-lo)//step)+1
            #     delta=random.randint(-2,2)*step
            #     li["n_heads"]=max(lo,min(hi, li["n_heads"]+delta))
            # # float
            # if random.random()<p_layer_float:
            #     s=self.layer_spec["mlp_ratio"]; lo,hi=s["low"],s["high"]
            #     sigma=(hi-lo)*0.05
            #     li["mlp_ratio"]=max(lo,min(hi, li["mlp_ratio"]+random.gauss(0,sigma)))
            # # categorical
            # if random.random()<p_layer_cat:
            #     choices=self.layer_spec["attn_type"]["choices"]
            #     cur=li["attn_type"]
            #     li["attn_type"]=random.choice([c for c in choices if c!=cur] or choices)

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
        min_layers = self.L_min
        if sum(mask) < min_layers:
            for _ in range(min_layers - sum(mask)):
                mask[random.randrange(self.L_max)] = True
        y["globals"]["layer_mask"] = mask

        p_rotate_layers = 0.1
        p_mirror_layers = 0.1
        # add the random rotation and mirror operations (Dihedral group D_L)
        if random.random() < p_rotate_layers and self.L_max >= 2:
            # rotation
            k = random.randint(1, self.L_max - 1)
            y["layers"] = y["layers"][k:] + y["layers"][:k]
            y["globals"]["layer_mask"] = y["globals"]["layer_mask"][k:] + y["globals"]["layer_mask"][:k]

        # mirroring
        if random.random() < p_mirror_layers and self.L_max >= 2:
            y["layers"] = y["layers"][::-1]
            y["globals"]["layer_mask"] = y["globals"]["layer_mask"][::-1]

        return self.repair(y)

    # def estimate_params_hetero(x):
    #     g = x["globals"]; d = g["d_model"]; seq_len = g["block_size"]
        
    #     # Embedding table parameters (same as GPT-2)
    #     vocab_size = 50257  # GPT-2 vocabulary size
    #     embedding_params = vocab_size * d  # token embeddings: vocab_size x d_model
    #     # embedding_params += seq_len * d    # positional embeddings: block_size x d_model
        
    #     total = embedding_params
    #     mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
    #     for i,active in enumerate(mask):
    #         if not active: continue
    #         li = x["layers"][i]
    #         h = max(1, li["n_heads"])
    #         r = li["mlp_ratio"]
    #         # very rough: per-layer params ~ 2*d^2 (proj) + r*d^2 (MLP) + heads overhead
    #         total += (2.0 + r)*d*d + 0.05*h*d*d
    #     return int(total)

    # def estimate_flops_hetero(x):
    #     g = x["globals"]; d = g["d_model"]; seq = g["block_size"]
    #     cost = 0.0
    #     mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
    #     for i,active in enumerate(mask):
    #         if not active: continue
    #         li = x["layers"][i]
    #         attn = li["attn_type"]
    #         attn_cost = {"scaled_dot": d*seq,
    #                     "gqa": d*seq/2,
    #                     "mha": d*math.log2(max(2, seq)),
    #                     "flash": d*seq*0.7}.get(attn, d*seq)
    #         cost += (2*d*d + attn_cost) + li["mlp_ratio"]*d*d
    #     return cost
    
    def calculate_possible_configs(self) -> int:
        total = 1
        for s in self.globals.values():
            if s["type"] == "int":
                step = s.get("step", 1)
                count = ((s["high"] - s["low"]) // step) + 1
                total *= count
            elif s["type"] == "float":
                # Assuming a reasonable discretization for floats
                total *= round(s["high"] - s["low"]) / 0.1  # arbitrary choice for float discretization
            elif s["type"] == "cat":
                total *= len(s["choices"])
        
        # Layer configurations
        layer_configs = 1
        for s in self.layer_spec.values():
            if s["type"] == "int":
                step = s.get("step", 1)
                count = ((s["high"] - s["low"]) // step) + 1
                layer_configs *= count
            elif s["type"] == "float":
                layer_configs *= round(s["high"] - s["low"]) / 0.1  # arbitrary choice for float discretization
            elif s["type"] == "cat":
                layer_configs *= len(s["choices"])
        
        total_layer_config = 0
        # Each layer can be active or inactive, except we enforce at least L_min active layers
        for layers_active in range(self.L_min, self.L_max + 1):
            total_layer_config += layer_configs * layers_active
        total *= total_layer_config
        return total


