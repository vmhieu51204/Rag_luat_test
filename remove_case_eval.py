import sys
import json
from pathlib import Path
from rag.evaluation.eval_utils import _aggregate, save_eval_results

def main():
    if len(sys.argv) != 3:
        print("Usage: python remove_case_eval.py <path_to_results.json> <doc_id_to_remove>")
        sys.exit(1)
        
    results_path = sys.argv[1]
    doc_id = sys.argv[2]
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    original_len = len(data.get("per_doc", []))
    
    # Remove the case by matching either doc_id or source_file
    new_per_doc = [
        doc for doc in data.get("per_doc", [])
        if doc.get("doc_id") != doc_id and doc.get("source_file") != doc_id
    ]
    
    if len(new_per_doc) == original_len:
        print(f"Case '{doc_id}' not found in {results_path}.")
        sys.exit(1)
        
    print(f"Removed case '{doc_id}'. Original count: {original_len}, New count: {len(new_per_doc)}")
    
    # Recalculate summary metrics using the utility function from eval_utils
    new_summary = _aggregate(new_per_doc)
    
    print("New Summary:")
    print(json.dumps(new_summary, indent=2))
    
    # Save the updated results back to the file
    save_eval_results(
        Path(results_path), 
        config=data.get("config", {}), 
        summary=new_summary, 
        per_doc=new_per_doc
    )
    print(f"Updated {results_path} successfully.")

if __name__ == "__main__":
    main()
