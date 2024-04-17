import re
import json
import random
import pymorphy3
import tokenize_uk
import pandas as pd
from translitua import translit
from src.constants import NAMES_PATH

morph = pymorphy3.MorphAnalyzer(lang='uk')

def detect_feminitive(text: str) -> bool:
    words = tokenize_uk.tokenize_uk.tokenize_words(text)
    verbs = [morph.parse(word)[0].word for word in words if morph.parse(word)[0].tag.POS == 'VERB']
    femn_verb_len = len([morph.parse(word)[0].tag.gender for word in verbs if morph.parse(word)[0].tag.gender == 'femn'])
    if femn_verb_len == 0 or len(verbs) == 0:
        return 0
    
    gender = femn_verb_len/len(verbs)
    return True if gender > 0.5 else False

def protected_groups_uk(text: str) -> bool:
    if re.search(r'сімейний статус', text) or re.search(r'заміжн', text) or re.search(r'одружен', text):
        return True
    if re.search(r'військов', text) or re.search(r'військ', text):
        return True
    if re.search(r'релігія', text):
        return True
    if re.search(r'мені \d{1,3} років', text.lower()):
        return True
    if re.search(r'жінк', text.lower()) or re.search(r'чоловік', text.lower()):
        return True
    
    names = load_names('uk', -1)
    if detect_name(text, names):
        return True
    return False

def protected_groups_en(text: str) -> bool:
    if re.search(r'marital status', text) or re.search(r'married', text):
        return True
    if re.search(r'military', text):
        return True
    if re.search(r'religion', text):
        return True
    if re.search(r'I am \d{1,3} years', text):
        return True
    if re.search(r'female', text) or re.search(r'male', text):
        return True
    
    names = load_names('en', -1)
    if detect_name(text, names):
        return True
    return False

def detect_name(text, names):
    words = tokenize_uk.tokenize_uk.tokenize_words(text)
    names = [word for word in words if word in names]
    if len(names) > 0:
        return True
    return False

def read_name_file(path: str) -> list:
    with open(path, 'r') as f:
        names = [ name.split()[0] for name in f.read().splitlines()]
    return names

def load_names(lang:str, n:int = 10, random_seed:int = 42) -> list:
    names = []
    for group_path in NAMES_PATH.values():
        group_name = read_name_file(group_path)
        if n != -1:
            random.seed(random_seed)
            random.shuffle(group_name)
            names.extend(group_name[:n//2])
        else:
            names.extend(group_name)
    if lang == 'en':
        names = [translit(name) for name in names]
    return names


def fix_decision_parser(df: pd.DataFrame) -> pd.DataFrame:
    if len(df[df['decision'].isna()]) > 0:
        ids = df[df['decision'].isna()].index
        for i in ids:
            try:
                if  "}" not in df.raw_ai_decision[i]:
                    parsed_answer = json.loads(df.raw_ai_decision[i]+'"}')
                else: 
                    parsed_answer = json.loads(df.raw_ai_decision[i].replace(",\n}","\n}"))

                df.loc[i, 'decision'] = parsed_answer['decision']
                df.loc[i, 'feedback'] = parsed_answer['feedback']
                df.loc[i, 'raw_ai_decision'] = json.dumps(parsed_answer)
            except:
                # TEMP SOLUTION
                parsed_answer = json.loads(df.raw_ai_decision[i].replace('"Сільпо"', "'Сільпо'").replace('"гарну English"', "'гарну English'")\
                                           .replace('"Profiles U"',"'Profiles U'").replace('"Just-Link-It"',"'Just-Link-It'")\
                                           .replace('"Whales Agency"', "'Whales Agency'").replace("```", ""))

                df.loc[i, 'decision'] = parsed_answer['decision']
                df.loc[i, 'feedback'] = parsed_answer['feedback']
                df.loc[i, 'raw_ai_decision'] = json.dumps(parsed_answer)
    return df
