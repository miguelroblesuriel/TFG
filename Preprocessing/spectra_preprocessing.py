def spectra_preprocessing(spectra):
    """Preprocess a list of spectra by applying default filters, normalizing intensities, and adding fingerprints.

    Args:
        spectra (list): List of spectra to preprocess.

    Returns:
        list: List of preprocessed spectra.
    """
    from matchms.filtering import default_filters, normalize_intensities, add_fingerprint

    processed_spectra = []
    for s in spectra:
        s = default_filters(s)           # cleans peaks, removes empty peaks
        s = normalize_intensities(s)     # normalize intensities to 0-1
        s = add_fingerprint(s)           # add fingerprint
        processed_spectra.append(s)

    return processed_spectra