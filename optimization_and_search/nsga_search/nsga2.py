import random, math, pickle
import os, time
from typing import Any, Dict, List
from search_space import HeteroSearchSpace, Individual
import hashlib, json, csv
from remote_trainer import RemoteTrainer
from hw_exp import evaluate_population
from hardware_search import evaluate_individual_on_hardware

# -----------------------------
# Problem with proxy evaluation
# -----------------------------
class EvaluationResult:
    def __init__(self, objs: List[float], cons: List[float], aux: Dict[str, Any]):
        self.objs = objs
        self.cons = cons
        self.aux = aux

class Population:
    # Holds individuals and their evaluations
    # initialized after evaluation 
    def __init__(self, individuals: List[Individual], evaluations: List[EvaluationResult] = None, search_space: HeteroSearchSpace = None, cons_settings: Dict[str, Any] = None, objs_settings: List[str] = None):
        self.individuals: List[Individual] = []
        for ind in individuals:
            if isinstance(ind, Individual):
                self.individuals.append(ind)
            elif isinstance(ind, dict):
                self.individuals.append(Individual.from_dict(ind))
            else:
                raise TypeError(f"Expected Individual or dict, got {type(ind)}")
        self.evaluations: List[EvaluationResult] = evaluations or []
        self.offspring: List[Individual] = []
        self.offspring_evaluations: List[EvaluationResult] = []
        self.gen = 0

        self.search_space = search_space
        self.objs_settings = objs_settings
        self.cons_settings = cons_settings

        # parameter options
        self.n_population = 16
        self.n_offspring = 8
        self.tournament_k = 2  # tournament selection size
        self.mutation_rate = 0.1  # mutation rate for offspring
        self.crossover_rate = 0.9  # crossover rate for offspring

    def print_summary(self):
        """Print a formatted summary of the population."""
        print(f"\n=== Population Summary (Generation {self.gen}) ===")
        print(f"Population size: {len(self.individuals)}")
        if self.offspring:
            print(f"Offspring size: {len(self.offspring)}")
        
        if self.evaluations:
            print(f"Evaluations completed: {len(self.evaluations)}")
            # Show objective statistics
            objs = [ev.objs for ev in self.evaluations]
            if objs:
                print(f"\nObjective Statistics:")
                # do not hardcode
                for i, obj_name in enumerate(self.objs_settings):
                    values = [o[i] for o in objs]
                    print(f"  {obj_name}: {min(values):.3f} - {max(values):.3f} (avg: {sum(values)/len(values):.3f})")

            # Show constraint violations
            cons = [ev.cons for ev in self.evaluations]
            if cons:
                print(f"\nConstraint Violations:")
                for i, con_name in enumerate(self.cons_settings.keys()):
                    values = [c[i] for c in cons]
                    violations = [v for v in values if v > 0]
                    print(f"  {con_name}: {len(violations)}/{len(values)} violated (max violation: {max(violations) if violations else 0:.3f})")
        else:
            print("No evaluations completed yet")
        
        print("=" * 50)

    def print_details(self):
        """Print detailed information of each individual and its evaluation."""
        for i, (ind, ev) in enumerate(zip(self.individuals, self.evaluations or [])):
            print(f"\n--- Individual {i+1} ---")
            indiv: Individual = ind
            indiv.print_individual()
            if ev:
                print(f"Objectives: {ev.objs}")
                print(f"Constraints: {ev.cons}")
                print(f"Auxiliary Info: {ev.aux}")
            else:
                print("No evaluation result available.")
        
        print("\n------------------------------------------")
        for i, ind in enumerate(self.offspring):
            print(f"\n--- Offspring Individual {i+1} ---")
            ind.print_individual()
            if self.offspring_evaluations and i < len(self.offspring_evaluations):
                ev = self.offspring_evaluations[i]
                print(f"Objectives: {ev.objs}")
                print(f"Constraints: {ev.cons}")
                print(f"Auxiliary Info: {ev.aux}")
            else:
                print("No evaluation result available.")

        print("\n" + "=" * 50)

    def __str__(self):
        """String representation of the population."""
        return f"Population(gen={self.gen}, size={len(self.individuals)}, evaluated={len(self.evaluations) if self.evaluations else 0})"

    def delete_duplicates(self):
        unique = {}
        for ind in self.individuals:
            key = json.dumps(ind, sort_keys=True)
            if key not in unique:
                unique[key] = ind
        self.individuals = list(unique.values())

    def fast_non_dominated_sort(self, objs: List[List[float]] = None, cons: List[List[float]] = None) -> List[List[int]]:
        """Perform non-dominated sorting and return a list of fronts (each front is a list of indices).

        If objs/cons are not provided, they will be derived from self.evaluations.
        - objs: List of objective vectors, each a list of floats (minimization).
        - cons: List of constraint vectors, each a list of floats (<= 0 is feasible).
        """
        # Derive from current evaluations if not explicitly given
        if objs is None or cons is None:
            if not self.evaluations:
                return []
            objs = [e.objs for e in self.evaluations]
            cons = [e.cons for e in self.evaluations]

        N = len(objs)
        S = [[] for _ in range(N)]
        n = [0] * N
        rank = [None] * N
        F: List[List[int]] = [[]]
        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                d = dominates(objs[p], cons[p], objs[q], cons[q])
                if d == 1:
                    S[p].append(q)
                elif d == -1:
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                F[0].append(p)
        i = 0
        while F[i]:
            Q: List[int] = []
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            F.append(Q)
        return F[:-1]

    def reorder_by_non_domination(self) -> None:
        """Reorder population in-place by Pareto fronts and crowding distance.

        Returns the index permutation applied (indices into the previous order).

        Ordering rule:
        1) Sort individuals into non-dominated fronts (F0, F1, ...)
        2) Within each front, sort by crowding distance descending (diversity preference)
        The final order is F0 (by CD), then F1 (by CD), etc.
        """
        if not self.evaluations:
            return []
        objs = [e.objs for e in self.evaluations]
        cons = [e.cons for e in self.evaluations]
        fronts = self.fast_non_dominated_sort(objs, cons)
        order: List[int] = []
        for front in fronts:
            if not front:
                continue
            cd = crowding_distance(front, objs)
            order.extend(sorted(front, key=lambda i: cd[i], reverse=True))
        # Apply permutation to individuals and evaluations
        self.individuals = [self.individuals[i] for i in order]
        self.evaluations = [self.evaluations[i] for i in order]
        print(f"Reordered individuals and evaluations by non-domination: {order}")
        return
    
    def to_yaml(self, save_path: str = None) -> str:
        """Convert population to YAML format for training experiments.
        Only includes active layers based on layer_mask.
        """
        yaml_lines = ["# Example YAML configuration file for training experiments"]
        yaml_lines.append("# Generated from NSGA-II population")
        yaml_lines.append("# Note: n_layers is automatically determined from the length of n_head_layerlist")
        yaml_lines.append("")

        for i, individual in enumerate(self.individuals if self.gen == 0 else self.offspring):
            g = individual["globals"]
            layers = individual["layers"]
            mask = g.get("layer_mask", [True] * len(layers))
            
            # Get active layers only
            active_indices = [j for j, active in enumerate(mask) if active]
            
            if not active_indices:  # Skip if no active layers
                continue

            # Format YAML entry
            yaml_lines.append(f"- idx: {i+1}")
            
            # add the configs in global settings
            for key, value in g.items():
                if key != "layer_mask":  # Exclude layer_mask from globals
                    yaml_lines.append(f"  {key}: {value}")

            # Build lists for active layers
            for key, _ in layers[0].items():
                layerlist = []
                # if key ends with "_exp", convert to actual value
                if key.endswith("_exp"):
                    arg = f"{key[:-4]}_layerlist"
                    for j in active_indices:
                        if j < len(layers):
                            layer = layers[j]
                            layerlist.append(2 ** layer.get(key))
                else:
                    arg = f"{key}_layerlist"
                    for j in active_indices:
                        if j < len(layers):
                            layer = layers[j]
                            layerlist.append(layer.get(key))
                yaml_lines.append(f"  {arg}: {layerlist}")

            yaml_lines.append(f"# n_layers: {len(active_indices)}")
            
            yaml_lines.append("")

        yaml_output = "\n".join(yaml_lines)
        file_name = f"{save_path or 'population'}/gen{self.gen}.yaml"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(yaml_output)
        
        return file_name

    def save_checkpoint(self, path: str) -> str:
        """Save a checkpoint of the population to JSON.

        Contents: gen (int), individuals, evaluations, offspring, offspring_evaluations,
        search_space config, and all population parameters.
        Writes atomically via a temporary file then rename.
        Returns the final path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "gen": int(self.gen),
            "individuals": self.individuals,
            "evaluations": None if self.evaluations is None else [
                {"objs": ev.objs, "cons": ev.cons, "aux": ev.aux} for ev in self.evaluations
            ],
            "offspring": self.offspring,
            "offspring_evaluations": None if self.offspring_evaluations is None else [
                {"objs": ev.objs, "cons": ev.cons, "aux": ev.aux} for ev in self.offspring_evaluations
            ],
            # Population parameters
            "n_offspring": self.n_offspring,
            "tournament_k": self.tournament_k,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            # Search space configuration (if available)
            "search_space_config": None if self.search_space is None else {
                "L_max": getattr(self.search_space, 'L_max', None),
                "d_model_choices": getattr(self.search_space, 'd_model_choices', None),
                "block_size_choices": getattr(self.search_space, 'block_size_choices', None),
                # Add other search space attributes as needed
            }
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return path

    def save_checkpoint_pkl(self, path: str) -> None:
        """Save a checkpoint of the population to a pickle file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return

    @staticmethod
    def load_checkpoint(path: str, from_pkl=True) -> "Population":
        """Load a Population from a checkpoint created by save_checkpoint.

        Note: EvaluationResult objects are reconstructed from stored dicts.
        Search space must be re-initialized separately if needed for operations.
        """
        if from_pkl:
            with open(path, "rb") as f:
                pop = pickle.load(f)
            if not isinstance(pop, Population):
                raise TypeError(f"Loaded object is not a Population, got {type(pop)}")
            return pop
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Load basic data
        individuals = data.get("individuals", [])
        
        # Load evaluations
        evals_raw = data.get("evaluations")
        evaluations = None
        if evals_raw is not None:
            evaluations = [EvaluationResult(er["objs"], er["cons"], er.get("aux", {})) for er in evals_raw]
        
        # Load offspring
        offspring = data.get("offspring", [])
        
        # Load offspring evaluations
        offspring_evals_raw = data.get("offspring_evaluations")
        offspring_evaluations = None
        if offspring_evals_raw is not None:
            offspring_evaluations = [EvaluationResult(er["objs"], er["cons"], er.get("aux", {})) for er in offspring_evals_raw]
        
        # Create population
        pop = Population(individuals, evaluations)
        pop.gen = int(data.get("gen", 0))
        pop.offspring = offspring
        pop.offspring_evaluations = offspring_evaluations if offspring_evaluations is not None else []
        
        # Restore population parameters
        pop.n_offspring = data.get("n_offspring", 32)
        pop.tournament_k = data.get("tournament_k", 2)
        pop.mutation_rate = data.get("mutation_rate", 0.1)
        pop.crossover_rate = data.get("crossover_rate", 0.9)
        
        # Note: search_space is not restored as it requires re-initialization
        # User should call pop.search_space = HeteroSearchSpace(...) after loading if needed
        
        return pop
    
    def hw_eval(self) -> list:
        # call hw_eval function from hw_exp.py
        if self.gen == 0:
            individuals = self.individuals
        else:
            individuals = self.offspring
        hw_data = evaluate_population(individuals, base_work_dir="./hw_eval/runs/")

        return hw_data

    def sw_eval(self, hosts: List[str], user: str, key_filename: str, run_dir_name: str, max_iters: int = 10000, conda_env: str = "reallmforge", sw_only: bool = False, hw_eval_on_reallmasic: bool = False) -> None:
        # send the training work to worker nodes and wait for results
        train_yaml_path = self.to_yaml(save_path="train")
        trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
        trainer.submit_job(path_to_yaml=train_yaml_path, remote_work_dir=f"/home/{user}/Evo_GPT", dir_name=run_dir_name, max_iters=max_iters, conda_env=conda_env)
        time.sleep(5)  # wait a bit before polling
        trainer.poll_jobs() 
        # start hw eval while waiting for training
        if sw_only:
            if hw_eval_on_reallmasic:
                hw_data = [evaluate_individual_on_hardware(ind) for ind in (self.individuals if self.gen == 0 else self.offspring)]
            else:
                hw_data = []
        else:
            start_time = time.time()
            hw_data = self.hw_eval()
            elapsed_time = time.time() - start_time
            print(f"Finished HW evaluation for generation {self.gen} in {elapsed_time:.1f}s")
            print("====================================================")
        trainer.wait_for_all(poll_interval=120, timeout=10000, verbose=True)
        data_csv = trainer.fetch_results(local_dir="train", gen=self.gen)
        # read the csv and populate self.evaluations
        # load the csv file's second column as a list of floats
        sw_data = load_csv_with_idx_lookup(data_csv)
        print (f"Loaded {len(sw_data)} results from {data_csv}")

        if self.gen == 0:
            self.evaluations = []
        else:
            self.offspring_evaluations = []

        if (self.objs_settings is None): 
            # set val_loss and params as default objectives
            self.objs_settings = ["val_loss", "params"]
        if (self.cons_settings is None):
            # set default constraints
            self.cons_settings = {
                "params": 800_000_000,  # 800 million params
                "val_loss": 3.6,  # 3.6 
            }

        if sw_only:
            print("Software-only evaluation completed.")
            # just aggregate sw data (optionally merge HW metrics if provided)
            for i, ind in enumerate(self.individuals if self.gen == 0 else self.offspring):
                sw_res = sw_data.get(i+1, float("inf"))
                params = ind.estimate_params()
                mem_bytes = ind.estimate_mem_access()
                flops = ind.estimate_flops()
                kv_cache_size = ind.estimate_kv_cache_size()
                hw_res = hw_data[i] if hw_data and i < len(hw_data) and hw_data[i] else {}
                auxs = {
                    "val_loss": sw_res,
                    "params": params/1e6,
                    "mem_bytes": mem_bytes,
                    "flops": flops/1e3,
                    "kv_cache_size": kv_cache_size/1e6,
                    **hw_res,
                }

                # Ensure all objectives/constraints exist; fall back to +inf if missing
                for key in self.objs_settings:
                    if key not in auxs or auxs[key] is None:
                        auxs[key] = float("inf")
                for key in self.cons_settings.keys():
                    if key not in auxs or auxs[key] is None:
                        auxs[key] = float("inf")

                objs = [float(auxs[obj]) for obj in self.objs_settings]
                cons = [float(auxs[con]) - self.cons_settings[con] for con in self.cons_settings.keys()]

                eval_res = EvaluationResult(objs, cons, auxs)
                if self.gen == 0:
                    self.evaluations.append(eval_res)
                else:
                    self.offspring_evaluations.append(eval_res)

                ind.print_individual()
                print(f"gen {self.gen} individual {i+1}: objs={objs}, cons={cons}, auxs={auxs}")
        else:
            self.aggregate_hw_sw_eval(sw_data, hw_data)


        return
    
    def aggregate_hw_sw_eval(self, sw_data: list, hw_data: list) -> None:
        """Aggregate software and hardware evaluation results."""
        if not sw_data or not hw_data:
            raise ValueError("Both SW and HW data must be provided.")
        
        # gather a list of metrics
        metrics = list(hw_data[0].keys()) + ["val_loss", "mem_bytes", "flops", "params"]

        # check if the chosen objs and cons are in the metrics
        for obj in self.objs_settings:
            if obj not in metrics:
                raise ValueError(f"Objective '{obj}' not found in evaluation metrics.")
        for cons in self.cons_settings.keys():
            if cons not in metrics:
                raise ValueError(f"Constraint '{cons}' not found in evaluation metrics.")
            
        # aggregate evaluations
        for i, ind in enumerate(self.individuals if self.gen == 0 else self.offspring):
            sw_res = sw_data[i+1] # idx in csv starts from 1
            hw_res = hw_data[i]
            params = ind.estimate_params()
            mem_bytes = ind.estimate_mem_access()
            flops = ind.estimate_flops()
            auxs = {
                "val_loss": sw_res,
                "params": params,
                "mem_bytes": mem_bytes,
                "flops": flops,
                **{k: hw_res[k] for k in hw_res.keys()}
            }
            objs = [float(auxs[obj]) for obj in self.objs_settings]
            cons = [float(auxs[con]) - self.cons_settings[con] for con in self.cons_settings.keys()]

            eval_res = EvaluationResult(objs, cons, auxs)
            if self.gen == 0:
                self.evaluations.append(eval_res)
            else:
                self.offspring_evaluations.append(eval_res)

            ind.print_individual()
            print(f"gen {self.gen} individual {i+1}: objs={objs}, cons={cons}, auxs={auxs}")
        return

    def generate_offspring(self) -> None:
        """Generate offspring via tournament selection and mutation."""
        if self.evaluations is None or not self.evaluations:
            raise ValueError("Cannot generate offspring without evaluations.")
        if self.search_space is None:
            raise ValueError("Search space is not defined for mutation.")
        search_space = self.search_space
        offspring = []
        for _ in range(self.n_offspring):
            p1_idx = tournament_select(self.individuals, self.evaluations, k=self.tournament_k)
            p2_idx = tournament_select(self.individuals, self.evaluations, k=self.tournament_k)
            parent1, _ = search_space.crossover(self.individuals[p1_idx], self.individuals[p2_idx], self.crossover_rate)
            child1 = self.search_space.mutate(parent1, self.mutation_rate)
            # child2 = self.search_space.mutate(parent2, self.mutation_rate)
            print("Generated offspring:")
            child1.print_individual()
            offspring.append(child1)

        self.offspring = offspring
        self.offspring_evaluations = []
        self.gen += 1
        print(f"Generated {self.n_offspring} offspring for generation {self.gen}")
        return
    
    def generate_offspring_random(self) -> None:
        """Generate offspring randomly from the search space."""
        if self.search_space is None:
            raise ValueError("Search space is not defined for sampling.")
        offspring = []
        for _ in range(self.n_offspring):
            child = self.search_space.sample()
            print("Generated random offspring:")
            child.print_individual()
            offspring.append(child)

        self.offspring = offspring
        self.offspring_evaluations = []
        self.gen += 1
        print(f"Generated {self.n_offspring} random offspring for generation {self.gen}")
        return

    def update_elimination(self, verbose: bool = False) -> None:
        if self.offspring_evaluations is None or not self.offspring_evaluations:
            raise ValueError("Cannot update elimination without offspring evaluations.")

        # append offspring to current population
        self.individuals.extend(self.offspring)
        self.evaluations.extend(self.offspring_evaluations)

        # Clear the offspring lists for the next generation
        self.offspring = []
        self.offspring_evaluations = []

        # Reorder by non-domination and keep the best individuals
        self.reorder_by_non_domination()
        if len(self.individuals) > self.n_population:
            print(f"Eliminating {len(self.individuals) - self.n_population} individuals to maintain population size {self.n_population}.")
            self.individuals = self.individuals[:self.n_population]
            self.evaluations = self.evaluations[:self.n_population]
        else:
            print(f"Population size {len(self.individuals)} is within limit {self.n_population}, no elimination needed.")

        if verbose:
            print(f"After elimination, population size: {len(self.individuals)}")
            for i, (ind, ev) in enumerate(zip(self.individuals, self.evaluations)):
                print(f"Individual {i+1}: Objs={ev.objs}, Cons={ev.cons}, Aux={ev.aux}")
        return

# -----------------------------
# CSV loading utility
# -----------------------------
def load_csv_with_idx_lookup(filepath):
    """Load CSV and return a dict for idx-based lookup."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['#idx'])
            best_val_loss = float(row[' best_val_loss'])
            data[idx] = best_val_loss
    return data

def estimate_params_hetero(x: Individual):
    g = x["globals"]; d = g["d_model"]
    total = 0.0
    mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
    indices = [i for i,active in enumerate(mask) if active]
    for i in indices:
        li = x["layers"][i]
        h = max(1, li["n_heads"])
        r = li["mlp_ratio"]
        total += (2.0 + r) * d * d + 0.03 * h * d * d
    return int(total)

def estimate_flops_hetero(x: Individual):
    g = x["globals"]; d = g["d_model"]; seq = g["block_size"]
    cost = 0.0
    mask = x["globals"].get("layer_mask", [True]*len(x["layers"]))
    indices = [i for i,active in enumerate(mask) if active]
    for i in indices:
        li = x["layers"][i]
        attn = li["attn_type"]
        attn_cost = {"scaled_dot": d*seq,
                     "gqa": d*seq/2,
                     "mha": d*math.log2(max(2, seq)),
                     "flash": d*seq*0.7}.get(attn, d*seq)
        cost += (2*d*d + attn_cost) + li["mlp_ratio"]*d*d
    return cost

def estimate_mem_hetero(x: Individual):
    params = estimate_params_hetero(x)
    bytes_per_param = max(1, x["globals"]["quant_bits"] // 8)
    seq = x["globals"]["block_size"]; d = x["globals"]["d_model"]
    # KV cache proxy (rough): 4 * seq * d bytes
    kv = 4 * seq * d
    return int(params*bytes_per_param + kv)

# -----------------------------
# NSGA-II core
# -----------------------------
def dominates(o1, c1, o2, c2):
    feas1 = all(c <= 0 for c in c1)
    feas2 = all(c <= 0 for c in c2)
    if feas1 and not feas2: return 1
    if feas2 and not feas1: return -1
    if feas1 and feas2:
        better = False; worse = False
        for a,b in zip(o1,o2):
            if a < b - 1e-12: better = True
            elif a > b + 1e-12: worse = True
        if better and not worse: return 1
        if worse and not better: return -1
        return 0
    v1 = sum(max(0.0, c) for c in c1)
    v2 = sum(max(0.0, c) for c in c2)
    if v1 < v2 - 1e-12: return 1
    if v1 > v2 + 1e-12: return -1
    return 0

def fast_non_dominated_sort(objs, cons):
    N = len(objs)
    S = [[] for _ in range(N)]
    n = [0]*N
    rank = [None]*N
    F = [[]]
    for p in range(N):
        for q in range(N):
            if p==q: continue
            d = dominates(objs[p], cons[p], objs[q], cons[q])
            if d == 1: S[p].append(q)
            elif d == -1: n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            F[0].append(p)
    i = 0
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        F.append(Q)
    return F[:-1]

def crowding_distance(front, objs):
    if not front: return {}
    # verify objs are of the same len
    n_objs = len(objs[0])
    for i, obj in enumerate(objs):
        if len(obj) != n_objs:
            raise ValueError(f"Objective length mismatch at index {i}: got {len(obj)}, expected {n_objs}")
        if any(x is None or (isinstance(x, float) and (x != x)) for x in obj):
            raise ValueError(f"Invalid objective value (None/NaN) at index {i}: {obj}")

    m = len(objs[0])
    dist = {i: 0.0 for i in front}
    for k in range(m):
        front_sorted = sorted(front, key=lambda i: objs[i][k])
        fmin = objs[front_sorted[0]][k]
        fmax = objs[front_sorted[-1]][k]
        dist[front_sorted[0]] = float('inf')
        dist[front_sorted[-1]] = float('inf')
        denom = (fmax - fmin) if abs(fmax - fmin) > 1e-12 else 1.0
        for i in range(1, len(front_sorted)-1):
            prev = objs[front_sorted[i-1]][k]
            nxt  = objs[front_sorted[i+1]][k]
            dist[front_sorted[i]] += (nxt - prev)/denom
    return dist

def tournament_select(pop, evals, k=2):
    i = random.randrange(len(pop))
    for _ in range(k-1):
        j = random.randrange(len(pop))
        if j == i: continue
        d = dominates(evals[i].objs, evals[i].cons, evals[j].objs, evals[j].cons)
        if d == -1 or (d == 0 and random.random() < 0.5):
            i = j
    return i

