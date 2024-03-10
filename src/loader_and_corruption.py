import json
import random
import logging
import datasets
import pandas as pd 
from src.helpers import protected_groups_uk, protected_groups_en, load_names
from src.constants import DATA_PATH, MATCHER_PATH, PRIMARY_POSITIONS, PROTECTED_GROUPS

logger = logging.getLogger("loader_and_corruption")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class DataLoader:
    """class for loading data and structure in proper way"""
    CANDIATES_PER_POSITION = 5
    JOBS_PER_CANDIDATE = 3

    def __init__(self, 
                 lang: str = 'uk',
                 random_state: int = 42,
                 dataset_lenght: int = 450):
        """
        Initialize DataLoader class
        
        Args:
            lang (str):           language of the data, default is 'uk'
            random_state (int):   random state for sampling, default is 42
            dataset_lenght (int): number of samples in the dataset, default is 450
            
        Returns:
            None
        """
        self.path_candidates = DATA_PATH[lang]["candidates"]
        self.path_jobs = DATA_PATH[lang]["jobs"]
        self.path_matchers = MATCHER_PATH
        self.lang = lang
        self.random_state = random_state
        self.dataset_lenght = dataset_lenght
        self.candidates_count = self.dataset_lenght/self.JOBS_PER_CANDIDATE

    def process(self, if_sampling: bool = True) -> pd.DataFrame:
        """
        Method for processing data
        
        Args:
            if_sampling (bool):   if True, sampling data, default is True
        
        Returns:
            pd.DataFrame:  processed data
        """
        logger.info('Loading data...')
        candidates = self._load_hf_dataset(self.path_candidates)
        jobs = self._load_hf_dataset(self.path_jobs)
        matchers = self._load_json(self.path_matchers)
        logger.info('Data loaded')

        logger.info('Filtering data...')
        candidates = self._data_filtering(candidates)
        logger.info('Data filtered')

        if if_sampling:
            logger.info('Sampling data...')
            candidates = self._data_sampling(candidates, matchers)
            print(candidates.shape[0])
            logger.info('Data sampled')

        logger.info('Combining data...')
        data = self._data_combining(candidates, jobs, matchers)
        logger.info('Data combined')
        return data

    def _data_sampling(self, candidates: pd.DataFrame, matchers: dict[str, str]) -> pd.DataFrame:
        """
        Method for sampling data
        
        Args:
            candidates (pd.DataFrame):   candidates data
            matchers (dict[str, str]):    matchers data
            
        Returns:
            pd.DataFrame:  sampled data
        """
        matchers = {k: v for k, v in matchers.items() if len(v) >= self.JOBS_PER_CANDIDATE}
        candidates = candidates[candidates['id'].isin(matchers.keys())]
        candidates.loc[:, 'Position match'] = candidates['Position'].apply(lambda x: x.lower() if self.lang == 'uk' else x)

        if len(candidates) < self.dataset_lenght:
            raise ValueError('Not enough candidates for sampling')

        # sample CANDIATES_PER_PRIMARY_KEYWORD candidates for each primary keyword if we do not have enough than get more in the next primary keyword   
        candidates_ids = []
        while len(candidates_ids) < self.candidates_count:
            for position in PRIMARY_POSITIONS if self.lang == 'en' else candidates['Position match'].value_counts().index:
                df_sample = candidates[candidates['Position match'] == position]
                if len(df_sample) < 5:
                    continue

                all_candidtes_ids  = df_sample['id'].unique()
                random.seed(self.random_state)
                random.shuffle(candidates_ids)
                i = 0
                for candidate_id in all_candidtes_ids:
                    if candidate_id not in candidates_ids:
                        candidates_ids.append(candidate_id)
                        i += 1
                    if len(candidates_ids) >= self.candidates_count or i >= self.CANDIATES_PER_POSITION:  
                        break
                if len(candidates_ids) >= self.candidates_count:
                    break
        return candidates[candidates['id'].isin(candidates_ids)].reset_index(drop=True)
    
    def _data_combining(self, candidates: pd.DataFrame, jobs: pd.DataFrame, matchers: dict[str, str]) -> pd.DataFrame:
        """
        Method for combining data
        
        Args:
            candidates (pd.DataFrame):   candidates data
            jobs (pd.DataFrame):          jobs data
            matchers (dict[str, str]):    matchers data
            
        Returns:
            pd.DataFrame:  combined data
        """
        data = []
        for candidate in candidates.to_dict('records'):
            if candidate['id'] in matchers.keys():
                job_ids = matchers[candidate['id']]
                random.seed(self.random_state)
                random.shuffle(job_ids)
                for job_id in job_ids[:self.JOBS_PER_CANDIDATE]:
                    data.append({
                        'item_id': candidate['id']+"_"+job_id,
                        'candidate_id': candidate['id'],
                        'job_id': job_id,
                        'CV': candidate['CV'],
                        'Job Description': jobs[jobs['id'] == job_id]['Long Description'].values[0],
                        'Job Position': jobs[jobs['id'] == job_id]['Position'].values[0],
                        'lang': candidate['CV_lang'],
                    })
        return pd.DataFrame(data)

    def _data_filtering(self, df: pd.DataFrame, column: str = 'CV') -> pd.DataFrame:
        """
        Method for filtering data

        Args:
            df (pd.DataFrame):   dataframe for filtering
            column (str):        column name for filtering, default is 'CV'
        
        Returns:
            pd.DataFrame:  filtered data
        """
        # filter data from protected groups
        if self.lang  == 'uk':
            df = df[~df[column].apply(protected_groups_uk)]
        elif self.lang == 'en':
            df = df[~df[column].apply(protected_groups_en)]
        else:
            raise ValueError('Language is not supported')
        return df.reset_index(drop=True)
    
    @staticmethod
    def _load_json(path: str) -> dict:
        """
        Method for loading json file

        Args:
            path (str):   path to the json file

        Returns:
            dict:  loaded data
        """
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    
    @staticmethod
    def _load_hf_dataset(path: str) -> pd.DataFrame:
        """
        Method for loading Hugging Face dataset

        Args:
            path (str):   path to the dataset in Hugging Face Hub
    
        Returns:
            pd.DataFrame:  loaded data
        """
        data = datasets.load_dataset(path)['train']
        return pd.DataFrame(data)


class DataCorruption:
    """class for corruption data for specific protected groups"""
    def __init__(self, lang: str = 'uk'):
        """
        Initialize DataCorruption class
        
        Args:
            lang (str): language of the data, default is 'uk'
            
        Returns:
            None
        """
        self.lang = lang

    def process(self, df: pd.DataFrame, protected_group: str) -> pd.DataFrame:
        """
        Method for processing data

        Args:
            df (pd.DataFrame):           data for processing
            protprotected_group (str):   protected group

        Returns:
            None
        """
        
        protected_attr = self.get_protected_attr(protected_group)
        data = self._corrupt_data(df, protected_group, protected_attr)
        return data

    def get_protected_attr(self, protected_group: str) -> list:
        """
        Method for getting protected attribute

        Args:
            protected_group (str):   protected group
        Returns:
            list:  protected attribute
        """
        if protected_group == "age":
            protected_attr = [20, 30, 40, 50, 60, 70]
        elif protected_group == "name":
            # 10 names: 5 males and 5 females
            protected_attr = load_names(self.lang, 10)
        else:
            protected_attr = self._load_txt(PROTECTED_GROUPS[protected_group][self.lang])

        return protected_attr
    
    def _corrupt_data(self, df: pd.DataFrame, protected_group: str, protected_attr: list) -> pd.DataFrame:
        """
        Method for corruption data

        Args:
            data (pd.DataFrame):         data for processing
            protected_group (str):       protected group
            protected_attr (list):       protected attribute

        Returns:
            pd.DataFrame:  corrupted data
        """
        data = []
        for group_item in df.to_dict('records'):
            for attr in protected_attr:
                group_item_copy = group_item.copy()
                group_item_copy['protected_group'] = protected_group
                group_item_copy['protected_attr'] = attr
                group_item_copy['group_id'] = group_item_copy['item_id']    
                del group_item_copy['item_id']
                data.append(group_item_copy)
        return pd.DataFrame(data)

    @staticmethod
    def _load_txt(path: str) -> list:
        """
        Method for loading txt file

        Args:
            path (str):   path to the txt file

        Returns:
            list:  loaded data
        """
        with open(path, 'r') as file:
            data = file.readlines()
        data = [item.strip() for item in data]
        return data