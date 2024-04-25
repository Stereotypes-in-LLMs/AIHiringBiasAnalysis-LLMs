import os
import time
import json
import logging
import pandas as pd

from loader_and_injection import DataInjection
from src.prompt import PROMPTS
from src.constants import PROTECTED_GROUPS_LIST_EN, PROTECTED_GROUPS_LIST_UK

logger = logging.getLogger("experiment_runner")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def experiment_core(corrupted_data: pd.DataFrame, corrupted_data_records: list, group_en: str, chain: object, save_root_path: str, data_paths: dict, batch_size: int = 32) -> dict:
    """
    Core method for running the experiment

    Args:
        corrupted_data (pd.DataFrame):       corrupted data
        corrupted_data_records (list):       corrupted data records
        group_en (str):                      protected group
        chain (object):                      chain Langchain object
        save_root_path (str):                path to save the results
        data_paths (dict):                   dictionary to store the paths
        batch_size (int):                    batch size for processing, default is 32

    Returns:
        dict:  dictionary with the paths
    """
    generated_data = []
    while len(generated_data) < len(corrupted_data):
        batch_data = corrupted_data_records[len(generated_data):len(generated_data)+batch_size]
        get_result = False 
        i = 0
        while (not get_result) and (i < 10):
            try:
                results = chain.batch(batch_data, config={"max_concurrency": batch_size})
                get_result = True
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(30)
                i += 1

        generated_data.extend(process_output(results))

        # save temp data
        with open(f"temp.jsonl", "w") as f:
            for item in generated_data:
                f.write(json.dumps(item) + "\n")

        if len(generated_data) % 500 == 0:
            logger.info(f"Generated {len(generated_data)} records")

    generated_decision = [val['decision'] if (isinstance(val, dict)) and ("decision" in val.keys()) else "" for val in generated_data]
    generated_feedback = [val['feedback'] if (isinstance(val, dict)) and ("feedback" in val.keys()) else "" for val in generated_data]

    corrupted_data["decision"] = generated_decision
    corrupted_data["feedback"] = generated_feedback
    corrupted_data["raw_ai_decision"] = generated_data
    logger.info(f"Saving {group_en}")
    corrupted_data.to_csv(os.path.join(save_root_path, f"{group_en}.csv"), index=False)
    data_paths[group_en] = os.path.join(save_root_path, f"{group_en}.csv")
    return data_paths

def run_experiment(folder_path: str,  chain: object, data: pd.DataFrame, lang: str, batch_size: int = 32, force_run: bool = False) -> dict:
    """
    Run experiment for all protected groups
    
    Args:
        folder_path (str):   path to store the results
        chain (object):      chain Langchain object
        data (pd.DataFrame): data to process
        lang (str):          language of the data
        batch_size (int):    batch size for processing, default is 32
        force_run (bool):    if True, force run the experiment, default is False
    """
    logger.info(f"Running experiment for {lang} and saving to {folder_path}.")
    data_paths = {}
    save_root_path = os.path.join(folder_path, lang)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    data_corruption = DataInjection(lang=lang) 

    for group_en, group_uk in zip(PROTECTED_GROUPS_LIST_EN, PROTECTED_GROUPS_LIST_UK):
        if not force_run and os.path.exists(os.path.join(save_root_path, f"{group_en}.csv")):    
            logger.info(f"Skipping {group_en}, already exists")
            continue
        logger.info(f"Running {group_en}")
        corrupted_data = data_corruption.process(data, group_en)
        corrupted_data_records = corrupted_data.to_dict(orient='records')
        corrupted_data_records = [{"job_desc": val["Job Description"], "candidate_cv": val["CV"], "protected_group": group_en if lang == "en" else group_uk, "protected_attr": val["protected_attr"]} for val in corrupted_data_records]

        data_paths = experiment_core(corrupted_data, corrupted_data_records, group_en, chain, save_root_path, data_paths, batch_size=batch_size)
    logger.info(f"Finished experiment for {lang} and saving to {folder_path}.")
    return data_paths

def run_experimment_second_model_verify(folder_path: str,  chain: object, based_on_results: str, lang: str, batch_size: int = 32, force_run: bool = False, test_id: list = None) -> dict:
    """
    Run experiment for all protected groups
    
    Args:
        folder_path (str):      path to store the results
        chain (object):         chain Langchain object
        based_on_results (str): path to the results of the first model
        lang (str):             language of the data
        batch_size (int):       batch size for processing, default is 32
        force_run (bool):       if True, force run the experiment, default is False
        test_id (list):         list of test ids to run the experiment. Default is None
    """
    logger.info(f"Running experiment for {lang} and saving to {folder_path}.")
    data_paths = {}
    save_root_path = os.path.join(folder_path, lang)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    if not os.path.exists(based_on_results):
        raise Exception(f"{based_on_results} folder path not found")
    
    for group_en, group_uk in zip(PROTECTED_GROUPS_LIST_EN, PROTECTED_GROUPS_LIST_UK):
        if not force_run and os.path.exists(os.path.join(save_root_path, f"{group_en}.csv")):    
            logger.info(f"Skipping {group_en}, already exists")
            continue
        corrupted_data = pd.read_csv(os.path.join(based_on_results, lang, f"{group_en}.csv"))
        if test_id is not None:
            corrupted_data = corrupted_data[corrupted_data['group_id'].isin(test_id)]
        corrupted_data_records = corrupted_data.to_dict(orient='records')
        corrupted_data_records = [{
                                    "job_desc": val["Job Description"], 
                                    "candidate_cv": val["CV"], 
                                    "protected_group": group_en if lang == "en" else group_uk, 
                                    "protected_attr": val["protected_attr"],
                                    "decision": val["decision"],
                                    "feedback": val["feedback"]
                                } 
                                  for val in corrupted_data_records]
        data_paths = experiment_core(corrupted_data, corrupted_data_records, group_en, chain, save_root_path, data_paths, batch_size=batch_size)
    logger.info(f"Finished experiment for {lang} and saving to {folder_path}.")
    return data_paths

def process_output(generated_results: list[object]) -> list:
    """
    Method for processing the generated results

    Args:
        generated_results (list[object]):   list of generated results

    Returns:
        list:  processed results
    """
    processed_results = []
    for result in generated_results:
        try:
            res = "{"+ result.content.split("{")[-1].split("}")[0] + "}"
            processed_results.append(json.loads(res))
        except:
            processed_results.append(result.content)
    return processed_results