# Math Dataset Generator for LLM Training

The GOAL: Generate data sets consisting of correct equations of the form 'a*b = c' in specified groups. Hopefully we can include more groups as we see fit.  So far we have: integers under addition, cyclic groups (modular addition), permutation groups, and dihedral groups (symmetries of n-gons). 

## For gen_unique_math.py

This script generates unique, plain-text datasets of integer or modular arithmetic statements in the format `a+b=c` or `a*b=c` (if multiplication is desired). It is designed for creating synthetic training data for Large Language Models (LLMs).

The script operates in two distinct modes:
1. **Standard Arithmetic:** Standard integer addition within a specified range.
2. **Modular Arithmetic:** Finite field addition ($\mathbb{Z}_N$) where inputs are strictly bounded by the modulus.

## Sample Usage

```bash
python gen_unique_math.py --count 100 --modulus 12 --mult true
```

The next line creates a set of 1000 correct addition equations among integers (infty) which range from -13 to 24.
```bash
python gen_unique_math.py --count 1000 --modulus infty --min -13 --max 24 --mult false
```

## For gen_group_theoretic_data.py

This script generates a data set of correct equations in a specified group G which is one of: 'integers', 'cyclic', 'permutation', 'dihedral'.  

### Example use: 

The following example recovers a list of correct integer additions and multiplications as before with specifications of max and min ranges.

```bash
python gen_group_math.py --group integers --count 1000 --min 0 --max 50 --mult true 
```

The next example does modular arithmetic additions mod 12:

```bash
python gen_group_math.py --group cyclic --count 100 --n 12
```

The next example does S3 the permutation group on the letters '1,2,3'. 

```bash
python gen_group_math.py --group permutation --count 30 --n 3
```

The next example does D8, the symmetry group of the square: 

```bash
python gen_group_math.py --group dihedral --count 50 --n 4
```

### Permutation group

We encode permutations by 1-line notation. So 1324 is the permutation that sends 1 --> 1, 2 --> 3 3 --> 2 and 4 --> 4. Similarly 4312 is the permutation on the symbols 1,2,3,4 that sends 1 --> 4, 2 -->3, 3 --> 1, 2 --> 4. 

### Dihedral group D2n

We encode the symmetries of the regular 5-gon (say) by a list of rotations r1, r2, ..., r5.  Then we list the right coset by a chosen flip s by the list r1s, r2s, ..., r5s.  These ten elements are the ten symmetries of a pentagon.





