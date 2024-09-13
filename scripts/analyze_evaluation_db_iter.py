import sqlite3
import json
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('with_det_960.db')
cursor = conn.cursor()

# Define the table name
table_name = 'data'

# Query to select all rows from the table
query = f'SELECT * FROM {table_name}'
cursor.execute(query)

# Fetch all rows from the table
rows = cursor.fetchall()

# Get the column names from the table
column_names = [description[0] for description in cursor.description]

# Iterate through each row and convert to JSON
cat_plans = dict()
all_plans = []
cat_results = dict()
all_results = []
for row in rows:
    # dict_keys(['id', 'image_id', 'question', 'answer', 'answer_gt', 'flag', 'info'])
    row_dict = dict(zip(column_names, row))
    info_obj = json.loads(row_dict['info'])
    label_type = info_obj['label_type']
    q = row_dict['question']

    evaluation_res = json.loads(row_dict['flag'])
    # is_correct = row_dict['answer'].strip() == row_dict['answer_gt'].strip()

    cat = label_type
    if (label_type == 'counting'):
        if ('building' in q):
            cat = 'counting_building'
        else:
            cat = 'counting'
    if (label_type == 'existence'):
        if ('building' in q):
            cat = 'existence_building'
        else:
            cat = 'existence'
    
    if (label_type not in ['seg', 'det']):
        # check ans
        if (cat not in cat_results):
            cat_results[cat] = []
        cat_results[cat].append(evaluation_res['ans'])
        all_results.append(evaluation_res['ans'])
        
    
    # always check plan
    if (cat not in cat_plans):
        cat_plans[cat] = []
    cat_plans[cat].append(evaluation_res['plan'])
    all_plans.append(evaluation_res['plan'])
    

print('all', round(np.mean(all_results), 6))
print({k: round(np.mean(cat_results[k]), 6) for k in cat_results})

print('all', round(np.mean(all_plans), 6))
print({k: round(np.mean(cat_plans[k]), 6) for k in cat_plans})

# Close the database connection
conn.close()
