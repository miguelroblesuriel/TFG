
from ms2deepscore.models import load_model
from matchms.Pipeline import Pipeline, create_workflow
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
from ms2deepscore import MS2DeepScore, SettingsMS2Deepscore
from ms2deepscore.wrapper_functions.training_wrapper_functions import train_ms2deepscore_wrapper
def ms2DeepScore_standard(file1,file2 = None):
    model_file_name = "ms2deepscore_model.pt"
    model = load_model(model_file_name)

    pipeline = Pipeline(create_workflow(query_filters=DEFAULT_FILTERS,
                                        score_computations=[[MS2DeepScore, {"model": model}]]))
    report = pipeline.run(file1)
    similarity_matrix = pipeline.scores.to_array()
    return similarity_matrix

def ms2DeepScore_trained(file1, file2 = None):
    settings = SettingsMS2Deepscore(
        additional_metadata=[("CategoricalToBinary", {"metadata_field": "ionmode",
                                                      "entries_becoming_one": "positive",
                                                      "entries_becoming_zero": "negative"}),
                             ("StandardScaler", {"metadata_field": "precursor_mz",
                                                 "mean": 0, "standard_deviation": 1000})], )
    train_ms2deepscore_wrapper(file1, settings, validation_split_fraction=20)
