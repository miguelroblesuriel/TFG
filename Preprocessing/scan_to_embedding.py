

def scan_to_embedding(filename):
    ms1_df, ms2_df = msql_fileloading.load_data(input_filepath, cache='feather')
    scans_filename = filename.replace(".mzML", ".npy")
    loaded_data = numpy.load(scans_filename, allow_pickle=True)  # importar los sets de dupla/tripletas
    scans = []
    embeddings = []
    for item in loaded_data:
        if item["triplet"] != []:
            scans.append(item["dupla"].tolist())
            scans.append(item["triplet"])
    for scan in scans:
        embedded_scans = [embedding['scan'] for embedding in embeddings]
        if scan not in embedded_scans:
            embedding = {
                'scan' : scan,
                'embedding' : create_embeddings(
                    precmz = ms2_df.loc[ms2_df['scan'] == scan, 'precursor_mz'].unique(),
                    mz = ms2_df.loc[ms2_df['scan'] == scan, 'mz'].values,
                    intensity = ms2_df.loc[ms2_df['scan'] == scan, 'i_norm'].values
                )
            }
            embeddings.append(embedding)
    return embeddings


