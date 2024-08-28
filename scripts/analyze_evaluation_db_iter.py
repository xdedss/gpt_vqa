import sqlite3
import json
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('simple_agent_rescuenet_valset_small_960_visualglm_untuned.db')
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
cat_results = dict()
all_results = []
for row in rows:
    # dict_keys(['id', 'image_id', 'question', 'answer', 'answer_gt', 'flag', 'info'])
    row_dict = dict(zip(column_names, row))
    info_obj = json.loads(row_dict['info'])
    label_type = info_obj['label_type']
    q = row_dict['question']

    is_correct = row_dict['flag'] == 'correct'
    # is_correct = row_dict['answer'].strip() == row_dict['answer_gt'].strip()

    cat = label_type
    if (label_type == 'seg'):
        continue
    if (label_type == 'counting'):
        # if (row_dict['answer_gt'] == '0'):
        #     continue
        if ('building' in q):
            cat = 'counting_building'
        else:
            cat = 'counting'
    if (label_type == 'existence'):
        if ('building' in q):
            cat = 'existence_building'
        else:
            cat = 'existence'
    
    
    if (cat not in cat_results):
        cat_results[cat] = []
    cat_results[cat].append(is_correct)
    all_results.append(is_correct)
    

print('all', round(np.mean(all_results), 6))
print({k: round(np.mean(cat_results[k]), 6) for k in cat_results})

# Close the database connection
conn.close()
