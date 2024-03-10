MATCHER_PATH = "../data/groups.json"
DATA_PATH = {
    "en": {
        "jobs": "Stereotypes-in-LLMs/recruitment-dataset-job-descriptions-english",
        "candidates": "Stereotypes-in-LLMs/recruitment-dataset-candidate-profiles-english",
    },
    "uk": {
        "jobs": "Stereotypes-in-LLMs/recruitment-dataset-job-descriptions-ukrainian",
        "candidates": "Stereotypes-in-LLMs/recruitment-dataset-candidate-profiles-ukrainian",
    }
}


PRIMARY_POSITIONS= [
    'QA Engineer', 
    'Project Manager', 
    'Front-end developer', 
    'Manual QA Engineer', 
    'UI/UX Designer', 
    'Java Developer', 
    'IT Recruiter', 
    'Python Developer', 
    'Business Analyst', 
    '.NET Developer', 
    'Data Analyst', 
    'iOS Developer', 
    'Full Stack Web Developer', 
    'React Developer', 
    'Graphic Designer', 
    'Software Engineer', 
    'DevOps engineer', 
    'Marketing Manager', 
    'Product manager', 
    'HR manager', 
    'Sales Manager',
    'PHP Developer', 
    'Android Developer', 
    'Node.js developer', 
    'Data Scientist', 
    'JavaScript Developer',
    'Automation QA Engineer', 
    '3D Artist', 
    'Product Designer', 
    'Web Developer'
]

PROTECTED_GROUPS_LIST_EN = ['gender', 'marital_status', 'military_status', 'religion', 'name', 'age']
PROTECTED_GROUPS_LIST_UK = ['стать', 'сімейний статус', 'військовий стантус', 'релігія', 'ім\'я', 'вік']
PROTECTED_GROUPS = {
    "gender": {
        "uk": "../protected_groups/gender_ukr.txt",
        "en": "../protected_groups/gender_en.txt"
    },
    "marital_status": {
        "uk": "../protected_groups/marital_status_ukr.txt",
        "en": "../protected_groups/marital_status_en.txt"
    },
    "military_status": {
        "uk": "../protected_groups/military_status_ukr.txt",
        "en": "../protected_groups/military_status_en.txt"
    },
    "religion": {
        "uk": "../protected_groups/religion_ukr.txt",
        "en": "../protected_groups/religion_en.txt"
    },
}

NAMES_PATH = {
    "female": '../protected_groups/ukr_names/fem_fname.txt',
    "male": '../protected_groups/ukr_names/masc_fname.txt'
}