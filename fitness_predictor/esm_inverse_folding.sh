ROOT_DATA_DIR="gym_single"
for fasta in $ROOT_DATA_DIR/fasta/*.fasta; do
    protein=$(basename $fasta .fasta)
    # Test whether the pdb file exists
    if [ ! -f $ROOT_DATA_DIR/alphafold2_pdb/$protein.pdb ]; then
        echo "PDB file gym_single/pdb/$protein.pdb does not exist"
        continue
    fi
    # Test whether the mutant file exists
    if [ ! -f $ROOT_DATA_DIR/mutant/$protein.csv ]; then
        echo "Mutant file gym_single/mutant/$protein.csv does not exist"
        continue
    fi
    # Score
    echo "Score the protein $protein"
    python fitness_predictor/esm_inverse_folding.py --fasta gym_single/fasta/$protein.fasta \
    --mutant gym_single/mutant/$protein.csv \
    --pdb gym_single/alphafold2_pdb/$protein.pdb
done

