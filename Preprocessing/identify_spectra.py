def identify_spectra(spectra1,spectra2):
    comparison_table = []
    i = 0
    for spec1 in spectra1:
        matches = []
        j = 0
        for spec2 in spectra2:
            if spec1.get("inchi") == spec2.get("inchi"):
                matches.append(j)
            j = j + 1
        comparison_table[i] = matches
        i = i + 1
    return comparison_table


