def get_greedy_hits(spectra, scores):
    hits = 0
    rank = 0
    for spec in spectra:
        sorted_matches = scores.scores_by_query(spec, name="CosineGreedy_score", sort=True)
        best_matches = [x for x in sorted_matches if x[1]["CosineGreedy_matches"] >= 0][:10]
        if spec.get("inchi") == best_matches[0][0].get("inchi"):
            hits += 1
            rank += 1
        else:
            i = 0
            for match in best_matches:
                i +=1
                if match[0].get("inchi") == spec.get("inchi"):
                    break
            rank += i
    rank = rank/len(spectra)
    return hits, rank