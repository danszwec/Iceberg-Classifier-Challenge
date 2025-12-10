import json
import pandas as pd
from typing import List, Tuple

def compare_ids() -> Tuple[bool, str]:
    """
    Compare IDs between sample_submission.csv and test.json to ensure alignment.
    
    Returns:
        Tuple[bool, str]: (True if aligned, message describing the comparison)
    """
    # Read sample_submission.csv IDs
    sample_df = pd.read_csv('sample_submission.csv')
    sample_ids = sample_df['id'].tolist()
    
    # Read test.json IDs
    with open('test.json', 'r') as f:
        test_data = json.load(f)
    test_ids = [entry['id'] for entry in test_data]
    
    # Compare counts
    if len(sample_ids) != len(test_ids):
        return False, f"Count mismatch: sample_submission has {len(sample_ids)} IDs, test.json has {len(test_ids)} IDs"
    
    # Compare IDs and order
    mismatches = []
    for i, (s_id, t_id) in enumerate(zip(sample_ids, test_ids)):
        if s_id != t_id:
            mismatches.append((i, s_id, t_id))
    
    if mismatches:
        mismatch_msg = f"Found {len(mismatches)} mismatches. First few: {mismatches[:5]}"
        return False, mismatch_msg
    
    # Check for duplicates
    if len(sample_ids) != len(set(sample_ids)):
        return False, "sample_submission.csv contains duplicate IDs"
    
    if len(test_ids) != len(set(test_ids)):
        return False, "test.json contains duplicate IDs"
    
    return True, f"✓ All {len(sample_ids)} IDs are aligned and in the same order"

if __name__ == '__main__':
    is_aligned, message = compare_ids()
    print(message)
    exit(0 if is_aligned else 1)

