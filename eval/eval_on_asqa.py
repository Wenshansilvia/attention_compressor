from datasets import load_dataset, Dataset
from metrics import (AnswerRougeCorrectness, AnswerEMCorrectness, AnswerDisambigF1Correctness)
import json

class asqa_bench():
    def __init__(self, dataset_path):
        self.dataset = load_dataset('json', data_files=dataset_path)['train']
        print(self.dataset)
        self.dataset = self.dataset.map(lambda example: {"short_answers": [s["short_answers"] for s in example["qa_pairs"]]})
        self.dataset = self.dataset.map(lambda example: {"long_answers": [s["long_answer"] for s in example["annotations"]]})
        self.dataset = self.dataset.rename_column('outputs', 'answers')
        self.batch_size = 10
        self.metrics = [AnswerRougeCorrectness(rouge_type="rougeL"), 
                AnswerEMCorrectness(ignore_case=True), 
                AnswerDisambigF1Correctness()]
        
    def evaluate(self):
        ground_truths = {
                "answer_disambig_f1": ("long_answers", "gt_answers"),
                "answer_rouge_correctness": ("long_answers", "gt_answers"),
                "answer_exact_match": ("short_answers", "gt_answers")
            }
        results = {}
        for m in self.metrics:
            print(f"Calculating {m.name}...")
            # Rename the ground truth column for metric calculation
            self.dataset = self.dataset.rename_column(*ground_truths[m.name])
            # Compute the metric
            results[m.name], self.dataset = m.compute(self.dataset, self.batch_size)
            # Rename the column back
            self.dataset = self.dataset.rename_column(*ground_truths[m.name][::-1])
        print(results)
        return results, self.dataset


if __name__ == '__main__':
    test_file_name = ''
    print(test_file_name)
    bench = asqa_bench(test_file_name)
    res, detail_res = bench.evaluate()
    print(res)
