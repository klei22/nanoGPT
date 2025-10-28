import os
import sys
import time
import timeloopfe.v4 as tl

# L = inputs cols
# E = inner dimension
# D = weights rows

def if_match_and_remove(flag, with_value = False):
    try:
        idx = sys.argv.index(flag)
        sys.argv.pop(idx)
        if with_value:
            return sys.argv.pop(idx)
        else:
            return True
        return
    except:
        return False

def parse_options():
    options = {
        "help": if_match_and_remove("-h") or if_match_and_remove("--help"),
        "live": if_match_and_remove("-l") or if_match_and_remove("--live"),
        "no_e_fusion": if_match_and_remove("-nef") or if_match_and_remove("--no_e_fusion"),
        "l_fusion": if_match_and_remove("-lf") or if_match_and_remove("--l_fusion"),
        "pay_writeback_in_matmul": if_match_and_remove("-pwim") or if_match_and_remove("--pay_wb_in_mm"),
        "convs": if_match_and_remove("-c") or if_match_and_remove("--convs")
    }
    return options

options = parse_options()

if options['help']:
    print("Available options:")
    print("-h, --help\t\tPrint this help menu.")
    print("-l, --live\t\tShow Timeloop's live status as it runs.")
    print("-nef, --no_e_fusion\tDo not enforce E=1 when doing fusions (useful only on matmul layers).")
    print("\t\t\t[Increases the latency with which column vectors are ready, due to DRAM accesses]")
    print("-lf, --l_fusion\t\tEnforce L=1 when doing fusions (useful only on matmul layers).")
    print("\t\t\t[Stores the entirety of the output in the Accumulator, useful if NO execution\n\t\t\tdoes not overlap with the matmul, but happens later]")
    print("-pwim, --pay_wb_in_mm\tWhen fusing, moves estimation of the cost of writing the final output from the NOs to the matmul.")
    print("\t\t\t[Essentially, now the matmul writes to DRAM, and the NO simply operates to and from on-chip memories]")
    print("-c, --convs\t\tWhen set, convolutions can be used as workloads.")
    sys.exit(0)

matmuls = ["KQV", "KTQ", "VScores", "Out", "FF1", "FF2"]
normops = ["softmax", "layernorm"]

convolutions_vgg = [f"vggL{i}" for i in range(16) if i not in (6, 9, 11, 12)] + ["vggL3+"]
convolutions_resnet = [f"resnetL{i}" for i in range(21) if i not in (2, 3, 4, 8, 9, 13, 14, 18, 19)] + ["resnetL1+", "resnetL3+"]

valid_args_1 = matmuls + normops + (convolutions_vgg + convolutions_resnet if options["convs"] else [])
desc_1 = [
    "First matmul, computes Q, K and V from Input with a projection.",
    "Second matmul, computes Scores as K^T*Q.",
    "Third matmul, computes V' with the convex combinations in V*Scores.",
    "Fourth matmul, computes Out from V' with a projection.",
    "Fifth matmul, first projection of the FF block, increases the latent dimension.",
    "Sixth matmul, second projection of the FF block, decreases back the latent dimension.",
    "Softmax operation, applied column-wise on the output of \"KTQ\".",
    "LayerNorm operation, applied column-wise on the output of \"Out\"."
] + (["<conv> see the respective layer file..." for _ in convolutions_vgg + convolutions_resnet] if options["convs"] else [])

# MUST STAY ORDERED! Set D,E=1 for all those above the specified one!
# No Scratchpad as it only stores inputs and outputs
memory_levels = ["DRAM", "Accumulator", "Registers"]
desc_memory_levels = [
    "DRAM, off-chip storage, fusing here is the same as doing nothing XD.",
    "Accumulator, on-chip SRAM storage dedicated to PEs outputs, usual target for fusions.",
    "Registers of the PEs, fusion possible iif all weights can fit them at once."
]

if len(sys.argv) not in [2, 3] or sys.argv[1] not in valid_args_1 or (len(sys.argv) == 3 and sys.argv[2] not in memory_levels):
    print("Error: Invalid argument(s).")
    print(f"Argvs: {sys.argv}")
    print("First argument must be one of the following:\n- {}".format('\n- '.join([a + ': ' + d for a, d in zip(valid_args_1, desc_1)])))
    print("Second argument can be the name of the memory hierarchy level at which to prepare for column-wise operation fusion. Hence one of the following:\n- {}".format('\n- '.join([a + ': ' + d for a, d in zip(memory_levels, desc_memory_levels)])))
    sys.exit(1)

print(f"Arguments provided: {sys.argv}")

is_norm_op = sys.argv[1] in normops
is_fusion = len(sys.argv) == 3
level_for_fusion = memory_levels.index(sys.argv[2]) if is_fusion else 0

# Define relative paths
ARCH_PATH = f"{os.curdir}/arch/system_gemmini{'_NOs' if is_norm_op else ''}.yaml"
COMPONENTS_PATH = f"{os.curdir}/arch/components/*.yaml"
PROBLEM_PATH = f"{os.curdir}/layers/{sys.argv[1]}_layer.yaml"
MAPPER_PATH = f"{os.curdir}/mapper/mapper.yaml"
CONSTRAINTS_PATH = f"{os.curdir}/constraints/constraints{'_NOs' if is_norm_op else ''}.yaml"
VARIABLES_PATH = f"{os.curdir}/mapper/variables.yaml"

if sys.argv[1] in convolutions_vgg:
    PROBLEM_PATH = PROBLEM_PATH.replace("/layers", "/layers/vgg")
elif sys.argv[1] in convolutions_resnet:
    PROBLEM_PATH = PROBLEM_PATH.replace("/layers", "/layers/resnet")

print(f"Sources:\n- {ARCH_PATH}\n- {COMPONENTS_PATH}\n- {MAPPER_PATH}\n- {PROBLEM_PATH}\n- {CONSTRAINTS_PATH}\n- {VARIABLES_PATH}\n")

spec = tl.Specification.from_yaml_files(
    ARCH_PATH,
    COMPONENTS_PATH,
    MAPPER_PATH,
    PROBLEM_PATH,
    CONSTRAINTS_PATH,
    VARIABLES_PATH
)  # Gather YAML files into a Python object

constrained_factors = ["D=1"]
if not options['no_e_fusion']: constrained_factors.append("E=1")
if options['l_fusion']: constrained_factors.append("L=1")
constrained_factors = tl.constraints.Factors(constrained_factors)

if spec.constraints['targets'] is None:
    spec.constraints['targets'] = tl.constraints.ConstraintsList()
if is_fusion:
    if not is_norm_op:
        # Apply fusion constraints
        found = []
        for target in spec.constraints['targets']:
            if target['target'] in memory_levels[0:level_for_fusion] and target['target'] not in found and target['type'] == "temporal":
                found.append(target['target'])
                target.factors = constrained_factors
                print(f"Updating constraint (factors): {dict(target.items())}")
            if (level_for_fusion >= 1 # fusing at Accumulator level
            and target['target'] == 'Scratchpad' # then remove inputs from the Scratchpad, as the Accumulator is lower down
            and target.type == "temporal"):
                target.factors = constrained_factors
                print(f"Updating constraint (factors): {dict(target.items())}")
        for target in list(filter(lambda x : x not in found, memory_levels[0:level_for_fusion])):
            spec.constraints['targets'].append(tl.constraints.constraint_factory({'target': target, 'type': 'temporal', 'factors': constrained_factors}))
            print(f"Adding constraint (factors): {dict(spec.constraints['targets'][-1].items())}")
        found.clear()

        if not options['pay_writeback_in_matmul']:
            for target in spec.constraints['targets']:
                if target['target'] in memory_levels[0:level_for_fusion] and target['target'] not in found and target['type'] == "dataspace":
                    found.append(target['target'])
                    # The cost of writing back outputs in in the NOs!
                    if "Outputs" not in target['bypass']:
                        target.bypass.append("Outputs")
                    if "Outputs" in target['keep']:
                        target.keep.remove("Outputs")
                    print(f"Updating constraint (bypasses): {dict(target.items())}")
            for target in list(filter(lambda x : x not in found, memory_levels[0:level_for_fusion])):
                spec.constraints['targets'].append(tl.constraints.constraint_factory({'target': target, 'type': 'dataspace', 'bypass': ["Outputs"], 'keep': ["Inputs", "Weights"]}))
                print(f"Adding constraint (bypasses): {dict(spec.constraints['targets'][-1].items())}")
            found.clear()

        # It is fine if outputs are written from the Accumulator to DRAM, as long as when there is
        # fusion full columns are available at the accumulator (enforce by forcing outer iterations
        # on E and D at 1). This is fine because during modelling of the NOs, when fusing at the
        # accumulator you SKIP the cost of writing outputs to DRAM! Thus, adding the two costs
        # (matmul and NO), you write to DRAM once anyway, but the NO adds the cost of the 3 passes
        # on the columns while they are in the accumulator!
        #for target in spec.constraints['targets']:
        #    if target['target'] in memory_levels[0:level_for_fusion] and target['target'] not in found and target.type == "dataspace":
        #        found.append(target['target'])
        #        if "Outputs" not in target['bypass']:
        #            target.bypass.append("Outputs")
        #        if "Outputs" in target['keep']:
        #            target.keep.remove("Outputs")
        #        print(f"Updating constraint: {dict(target.items())}")
        #for target in list(filter(lambda x : x not in found, memory_levels[0:level_for_fusion])):
        #    spec.constraints['targets'].append(tl.constraints.constraint_factory({'target': target, 'type': 'dataspace', 'keep': ["Inputs", "Weights"], 'bypass': ["Outputs"]}))
        #    print(f"Adding constraint: {dict(spec.constraints['targets'][-1].items())}")
        
        # THE LEVEL(s) AT WHICH YOU FUSE MUST HAVE LOOPS (inner)EDL(outer)
        for target in spec.constraints['targets']:
            if target['target'] in memory_levels[0:level_for_fusion+1] and target['target'] and target['type'] == "temporal":
                found.append(target['target'])
                target['permutation'] = tl.constraints.Permutation(['E', 'D', 'L'])
                print(f"Updating constraint (permutation): {dict(target.items())}")
                break
        for target in list(filter(lambda x : x not in found, memory_levels[0:level_for_fusion+1])):
            spec.constraints['targets'].append(tl.constraints.constraint_factory({'target': memory_levels[level_for_fusion], 'type': 'temporal', 'permutation': ['E', 'D', 'L']}))
            print(f"Adding constraint (permutation): {dict(spec.constraints['targets'][-1].items())}")
    
    if is_norm_op:
        print("WARNING: copying just the bypasses, disregarding latency between two columns being ready. This means that only energy is a valid estimate.")
        # Apply the fusion to the NO
        found = []
        for target in spec.constraints['targets']:
            if target['target'] in memory_levels[0:level_for_fusion] and target['target'] not in found and target['type'] == "dataspace":
                found.append(target['target'])
                if "Inputs" not in target['bypass']:
                    target.bypass.append("Inputs")
                if "Inputs" in target['keep']:
                    target.keep.remove("Inputs")
                if options['pay_writeback_in_matmul']:
                    if "Outputs" not in target['bypass']:
                        target.bypass.append("Outputs")
                    if "Outputs" in target['keep']:
                        target.keep.remove("Outputs")
                print(f"Updating constraint (bypasses): {dict(target.items())}")
            if (level_for_fusion >= 1 # fusing at Accumulator level
                and target['target'] == 'Scratchpad' # then remove inputs from the Scratchpad, as the Accumulator is lower down
                and target.type == "dataspace"):
                target.bypass.append("Inputs")
                target.keep.remove("Inputs")
                print(f"Updating constraint (bypasses): {dict(target.items())}")
        for target in list(filter(lambda x : x not in found, memory_levels[0:level_for_fusion])):
            spec.constraints['targets'].append(tl.constraints.constraint_factory({'target': target, 'type': 'dataspace', 'bypass': ["Inputs", "Weights", "Outputs"] if options['pay_writeback_in_matmul'] else ["Inputs", "Weights"]}))
            print(f"Adding constraint (bypasses): {dict(spec.constraints['targets'][-1].items())}")

if options['live']:
    spec.mapper.live_status = True

spec.mapspace.template = 'uber' #'ruby'

output_dir = f"{os.curdir}/outputs_{sys.argv[1]}" + (("_" + sys.argv[2]) if level_for_fusion else "")
if not os.path.exists(output_dir): os.makedirs(output_dir)
start_time = time.time()
tl.call_mapper(spec, output_dir=output_dir)  # Run the Timeloop mapper
print(f"\nTimeloop finished in: {time.time() - start_time:.3f}s")

#stats = open("outputs/timeloop-mapper.stats.txt").read()
#print(stats[stats.index("Summary Stats") :])

def read_and_indent(path, n_lines=30, start_at: str = None, end_at: str = None):
    content = open(path).read()
    if start_at is not None:
        content = content[content.index(start_at) :]
    if end_at is not None:
        content = content[: content.index(end_at) + len(end_at)]
    content = content.split("\n")
    content = content[:n_lines] if n_lines > 0 else content[n_lines:]
    return "\t" + "\n\t".join(content)

def append_to_file(path, text_to_append):
    with open(path, 'a') as file:
        file.write('\n\n' + text_to_append)

append_to_file(f"{output_dir}/timeloop-mapper.map.txt", f"Fusion optimized: {is_fusion}\nFusion constraints: {constrained_factors}")
print("\n\nMapping:")
print(read_and_indent(f"{output_dir}/timeloop-mapper.map.txt"))

