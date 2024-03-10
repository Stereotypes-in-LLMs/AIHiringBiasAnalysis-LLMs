import os
import json
import logging
import pandas as pd

from src.loader_and_corruption import DataCorruption
from src.prompt import PROMPTS
from src.constants import PROTECTED_GROUPS_LIST_EN, PROTECTED_GROUPS_LIST_UK

logger = logging.getLogger("experiment_runner")
logger.setLevel(logging.INFO)

def run_experiment(folder_path: str,  chain: object, data: pd.DataFrame, lang: str, batch_size: int = 32, force_run: bool = False):
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
    data_paths = {}
    save_root_path = os.path.join(folder_path, lang)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    data_corruption = DataCorruption(lang=lang) 

    for group_en, group_uk in zip(PROTECTED_GROUPS_LIST_EN, PROTECTED_GROUPS_LIST_UK):
        if not force_run and os.path.exists(os.path.join(save_root_path, f"{group_en}.csv")):    
            logger.info(f"Skipping {group_en}, already exists")
            continue
        logger.info(f"Running {group_en}")
        corrupted_data = data_corruption.process(data, group_en)
        corrupted_data_records = corrupted_data.to_dict(orient='records')
        corrupted_data_records = [{"job_desc": val["Job Description"], "candidate_cv": val["CV"], "protected_group": group_en if lang == "en" else group_uk, "protected_attr": val["protected_attr"]} for val in corrupted_data_records]

        generated_data = []
        while len(generated_data) < len(corrupted_data):
            batch_data = corrupted_data_records[len(generated_data):len(generated_data)+batch_size]
            results = chain.batch(batch_data, config={"max_concurrency": batch_size})

            generated_data.extend(process_output(results))

            # save temp data
            with open(f"temp.jsonl", "w") as f:
                for item in generated_data:
                    f.write(json.dumps(item) + "\n")

            if len(generated_data) % 500 == 0:
                logger.info(f"Generated {len(generated_data)} records")

        generated_decision = [val['decision'] if ("decision" in val.keys()) and (isinstance(val, dict)) else "" for val in generated_data]
        generated_feedback = [val['feedback'] if ("feedback" in val.keys()) and (isinstance(val, dict))  else "" for val in generated_data]

        corrupted_data["decision"] = generated_decision
        corrupted_data["feedback"] = generated_feedback
        corrupted_data["raw_ai_decision"] = generated_data
        logger.info(f"Saving {group_en}")
        corrupted_data.to_csv(os.path.join(save_root_path, f"{group_en}.csv"), index=False)
        data_paths[group_en] = os.path.join(save_root_path, f"{group_en}.csv")
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
            processed_results.append(json.loads(result.content))
        except:
            processed_results.append(result.content)
    return processed_results