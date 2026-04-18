from Comparison.cosine_greedy_spectra import cosine_greedy
from Comparison.createSpectrum import createSpectrum
from Preprocessing.spectra_preprocessing import spectra_preprocessing
import numpy

def compare_two_scans(scan1, scan2, ms2_df):
    spectra = []
    spectrum1 = createSpectrum(ms2_df[ms2_df['scan'] == scan1]['i'].to_numpy(),
                                  numpy.sort(ms2_df[ms2_df['scan'] == scan1]['mz'].to_numpy()), ms2_df[ms2_df['scan'] == scan1]['precmz'].unique(),scan1)
    spectrum2 = createSpectrum(ms2_df[ms2_df['scan'] == scan2]['i'].to_numpy(),
                                  numpy.sort(ms2_df[ms2_df['scan'] == scan2]['mz'].to_numpy()), ms2_df[ms2_df['scan'] == scan2]['precmz'].unique(),scan2)
    spectra.append(spectrum1)
    spectra.append(spectrum2)
    spectra = spectra_preprocessing(spectra)
    scores = cosine_greedy(0.005, spectra, spectra)
    return scores