# ProbTrim

ProbTrim is a PC-based tool for accurate adapter contamination detection from sequencing reads from human host. It uses a learned probabilistic models to infer and remove adapter sequences even in noisy or ambiguous reads.

## Requirements
- Python 3.8+
- PyTorch
- PyJuice
- BioPython
- NumPy
- Pandas
- tqdm

## How to use?

```python run.py --contaminated_path path/to/adapters.fasta --clean_path path/to/clean_sequences.fasta --eval_path path/to/reads_to_check_contam.fastq --output_path results/contamination_results.csv ``` 

