import os


def setup_paths():
    # This should be the project folder
    dir_path = os.getcwd()

    base_path = dir_path + "/"
    raw_data_path = base_path + "datasets/"
    cache_path = base_path + "cache/"
    interim_path = cache_path + "interim/"
    processed_path = cache_path + "processed/"
    output_path = cache_path + "output/"
    output_models_path = output_path + "models/"
    output_plots_path = output_path + "plots/"

    paths = {
        "base_path": base_path,
        "raw_data_path": raw_data_path,
        "cache_path": cache_path,
        "interim_path": interim_path,
        "processed_path": processed_path,
        "output_path": output_path,
        "output_models_path": output_models_path,
        "output_plots_path": output_plots_path,
    }

    for k, v in paths.items():
        if not os.path.isdir(v):
            print(v)
            os.makedirs(v)

    return paths
