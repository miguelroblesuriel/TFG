from pyopenms import *

def get_instrument_type(mzml_file):
    exp = MSExperiment()
    MzMLFile().load(mzml_file, exp)

    instrument = exp.getInstrument()

    instrument_name = instrument.getName()
    analyzers = []
    for analyzer in instrument.getMassAnalyzers():
        analyzers.append(analyzer.getType())

    return instrument_name, analyzers
