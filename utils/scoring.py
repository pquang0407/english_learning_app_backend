from difflib import SequenceMatcher

def score_transcription(pred: str, target: str):
    """
    Compares the speech recognition result (pred) with the target sentence (target)
    and returns a score, along with the status of each word in the *spoken* text.
    Uses difflib.SequenceMatcher to identify differences.
    """
    # Normalize inputs
    pred_tokens = pred.strip().upper().split()
    target_tokens = target.strip().upper().split()

    # Use SequenceMatcher to find matching words
    s = SequenceMatcher(None, target_tokens, pred_tokens)
    
    # We now build the matches list based on the spoken words (pred_tokens)
    matches = []
    
    # A set to keep track of correctly matched target words
    matched_target_indices = set()
    
    # Iterate through opcodes to get word status
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            # Words are correct, add them to the matches list
            for i in range(j1, j2):
                matches.append({"word": pred_tokens[i], "status": "correct"})
                matched_target_indices.add(i)
        elif tag == 'replace':
            # Words are incorrect, add them to the matches list
            for i in range(j1, j2):
                matches.append({"word": pred_tokens[i], "status": "wrong"})
        elif tag == 'insert':
            # Extra words added, add them to the matches list
            for i in range(j1, j2):
                matches.append({"word": pred_tokens[i], "status": "extra"})
        elif tag == 'delete':
            # Words were deleted from the spoken sentence (missing)
            pass

    # Count correct words for scoring
    correct_count = len([m for m in matches if m['status'] == 'correct'])
    
    # Calculate score based on the target
    total_target_words = len(target_tokens)
    if total_target_words > 0:
        score = int((correct_count / total_target_words) * 100)
    else:
        score = 0
    
    # Find missing words
    missing_words = [target_tokens[i] for i in range(total_target_words) if i not in matched_target_indices]

    return {
        "score": score,
        "matches": matches,
        "transcription": pred,
        "target": target,
        "total_words": total_target_words,
        "correct_words": correct_count,
        "missing_words": missing_words
    }
