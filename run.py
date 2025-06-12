import torch
import pyjuice as juice
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import random
import argparse
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AdapterDetectorCond:
    def __init__(self, window_size=50, stride=3, num_latents=128):
        self.window_size = window_size
        self.stride = stride
        self.num_latents = num_latents
        self.pc = None
        self.threshold = 0.8
        self.base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.num_base_cats = 5
        self.num_label_cats = 2
        
    def encode_sequence(self, seq, length=None):
        if length is None:
            length = self.window_size
        encoded = torch.full((length,), 4, dtype=torch.long)
        for i, base in enumerate(seq[:length]):
            if base.upper() in self.base_mapping:
                encoded[i] = self.base_mapping[base.upper()]
        return encoded
    
    def train_cond_data(self, contaminated_seqs, clean_seqs=None, balance_ratio=1.0):
        all_sequences = []
        all_labels = []
        
        contaminated_windows = []
        for seq in tqdm(contaminated_seqs):
            if len(seq) < self.window_size:
                window = seq.ljust(self.window_size, 'N')
                contaminated_windows.append(window)
            else:
                for i in range(0, len(seq) - self.window_size + 1, self.stride):
                    window = seq[i:i + self.window_size]
                    contaminated_windows.append(window)
        
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        for seq in tqdm(contaminated_seqs):
            rc_seq = ''.join(complement.get(base.upper(), 'N') for base in seq[::-1])
            if len(rc_seq) < self.window_size:
                window = rc_seq.ljust(self.window_size, 'N')
                contaminated_windows.append(window)
            else:
                for i in range(0, len(rc_seq) - self.window_size + 1, self.stride):
                    window = rc_seq[i:i + self.window_size]
                    contaminated_windows.append(window)
        
        for window in contaminated_windows:
            encoded_seq = self.encode_sequence(window)
            encoded_label = torch.tensor([1], dtype=torch.long)
            full_input = torch.cat([encoded_seq, encoded_label])
            all_sequences.append(full_input)
            all_labels.append(1)
        
        
        clean_windows = []
        for seq in tqdm(clean_seqs):
            if len(seq) < self.window_size:
                window = seq.ljust(self.window_size, 'N')
                clean_windows.append(window)
            else:
                for i in range(0, len(seq) - self.window_size + 1, self.stride):
                    window = seq[i:i + self.window_size]
                    clean_windows.append(window)
                    if len(clean_windows) >= len(contaminated_windows) * balance_ratio:
                        break
            if len(clean_windows) >= len(contaminated_windows) * balance_ratio:
                break
        
        for window in clean_windows:
            encoded_seq = self.encode_sequence(window)
            encoded_label = torch.tensor([0], dtype=torch.long)
            full_input = torch.cat([encoded_seq, encoded_label])
            all_sequences.append(full_input)
            all_labels.append(0)
        
        combined_data = list(zip(all_sequences, all_labels))
        random.shuffle(combined_data)
        
        sequences_tensor = torch.stack([item[0] for item in combined_data])
        labels_tensor = torch.tensor([item[1] for item in combined_data])
        
        return sequences_tensor, labels_tensor
    
    def build_conditional_pc(self, train_data):
        input_dim = self.window_size + 1
        block_size = 8
        
        ns = juice.structures.PD(
            data_shape=(self.window_size + 1,),  
            num_latents=self.num_latents,
            split_intervals=8,               
            input_node_params={'num_cats': max(self.num_base_cats, self.num_label_cats)},
            block_size=8
        )
        
        pc = juice.compile(ns)
        pc.to(device)
        return pc
    
    def train_cond(self, contaminated_seqs, clean_seqs=None, epochs=100, 
                         batch_size=256, learning_rate=0.1):
        train_data, labels = self.train_cond_data(contaminated_seqs, clean_seqs)
        
        train_dataset = TensorDataset(train_data, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        self.pc = self.build_conditional_pc(train_data)
        
        optimizer = juice.optim.CircuitOptimizer(
            self.pc, 
            lr=learning_rate, 
            pseudocount=0.1, 
            method="EM"
        )
        
        scheduler = juice.optim.CircuitScheduler(
            optimizer,
            method="multi_linear", 
            lrs=[learning_rate, learning_rate * 0.1, learning_rate * 0.01],
            milestone_steps=[0, len(train_loader) * epochs // 3, len(train_loader) * 2 * epochs // 3]
        )
        
        for epoch in range(1, epochs + 1):
            self.pc.train()
            epoch_train_ll = 0.0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                lls = self.pc(batch_data)
                loss = -lls.mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_train_ll += lls.mean().item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Train LL: {epoch_train_ll / len(train_loader):.4f}")
    
    def compute_contamination_score(self, sequence):
        
        if len(sequence) < self.window_size:
            sequence = sequence.ljust(self.window_size, 'N')
        
        max_log_ratio = float('-inf')
        best_position = 0
        
        num_windows = max(1, len(sequence) - self.window_size + 1)
        positions = list(range(0, num_windows, self.stride))
        
        batch_size = 8
        self.pc.eval()
        
        with torch.no_grad():
            for i in range(0, len(positions), batch_size):
                batch_positions = positions[i:i + batch_size]
                
                clean_batch = []
                contam_batch = []
                
                for pos in batch_positions:
                    window = sequence[pos:pos + self.window_size]
                    encoded_seq = self.encode_sequence(window)
                    
                    clean_input = torch.cat([encoded_seq, torch.tensor([0], dtype=torch.long)])
                    contam_input = torch.cat([encoded_seq, torch.tensor([1], dtype=torch.long)])
                    
                    clean_batch.append(clean_input)
                    contam_batch.append(contam_input)
                
                while len(clean_batch) < batch_size:
                    clean_batch.append(clean_batch[0])
                    contam_batch.append(contam_batch[0])
                
                clean_tensor = torch.stack(clean_batch).to(device)
                contam_tensor = torch.stack(contam_batch).to(device)
                
                ll_clean_batch = self.pc(clean_tensor)
                ll_contam_batch = self.pc(contam_tensor)
                
                for j, pos in enumerate(batch_positions):
                    ll_clean = ll_clean_batch[j].item()
                    ll_contam = ll_contam_batch[j].item()
                    
                    log_ratio = ll_contam - ll_clean
                    
                    if log_ratio > max_log_ratio:
                        max_log_ratio = log_ratio
                        best_position = pos
        
        contamination_score = 1 / (1 + np.exp(-max_log_ratio))
        
        return {
            'score': contamination_score,
            'log_ratio': max_log_ratio,
            'position': best_position
        }
    
    def detect_contamination(self, sequence, threshold=None):
            
        result = self.compute_contamination_score(sequence)
        
        if threshold == 0.0:
            result['contaminated'] = bool(result['log_ratio'] > 0.0)
        else:
            result['contaminated'] = bool(result['score'] > threshold)
            
        result['threshold'] = threshold
        return result

def load_sequences(fasta_path):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contaminated_path', required=True)
    parser.add_argument('--clean_path', required=True)
    parser.add_argument('--eval_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--file_format', default='fastq')
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--num_latents', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.8)
    
    args = parser.parse_args()

    detector = AdapterDetectorCond(
        window_size=args.window_size, 
        stride=args.stride, 
        num_latents=args.num_latents
    )

    contaminated_sequences = load_sequences(args.contaminated_path)
    clean_sequences = load_sequences(args.clean_path)

    detector.train_cond(
        contaminated_sequences,
        clean_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    results = []

    for read in tqdm(SeqIO.parse(args.eval_path, args.file_format)):
        result = detector.detect_contamination(str(read.seq), threshold=args.threshold)
        results.append({
            'id': read.id,
            'length': len(read.seq),
            'contamination_score': float(result['score']),
            'log_ratio': float(result['log_ratio']),
            'contaminated': bool(result['contaminated']),
            'position': int(result['position']) if result['position'] is not None else None,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()