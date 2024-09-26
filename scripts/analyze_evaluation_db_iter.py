import sqlite3
import json
import numpy as np
import re

# Connect to the SQLite database
conn = sqlite3.connect('with_det_1k_real.db')
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

class CatBoolMeter():

    def __init__(self):
        self.by_cat = dict()
        self.all_res = []
    
    def add(self, cat, b):
        if (cat not in self.by_cat):
            self.by_cat[cat] = []
        self.by_cat[cat].append(b)
        self.all_res.append(b)
    
    def print_stat(self):
        print('all:', round(np.mean(self.all_res), 6))
        print('by category:', {k: round(np.mean(self.by_cat[k]), 6) for k in self.by_cat})
    
    def get_mean(self):
        return np.mean(self.all_res)



# Iterate through each row and convert to JSON
cat_plans = CatBoolMeter()
cat_results = CatBoolMeter()
cat_exact_match = CatBoolMeter()
cat_plan_vr = CatBoolMeter()
cat_plan_r = CatBoolMeter()
cat_plan_p = CatBoolMeter()

for row in rows:
    # dict_keys(['id', 'image_id', 'question', 'answer', 'answer_gt', 'flag', 'info'])
    row_dict = dict(zip(column_names, row))
    info_obj = json.loads(row_dict['info'])
    label_type = info_obj['label_type']
    q = row_dict['question']
    gt = row_dict['answer_gt']
    answer = row_dict['answer']

    # print(row_dict['flag'])
    if (row_dict['flag'] == 'error'):
        evaluation_res = {'ans': False, 'plan': False}
    else:
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
    if (label_type == 'area'):
        if ('building' in q):
            cat = 'area_building'
        else:
            cat = 'area'
    
    if (label_type not in ['seg', 'det']):
        # check ans
        cat_results.add(cat, evaluation_res['ans'])

        regex = f'\\b{re.escape(gt.lower().strip())}'
        if (label_type in ['area_building', 'area']):
            try:
                last_float = float(re.findall(r'-?\d+\.?\d*', answer)[-1])
                exact_match = abs(last_float - float(gt)) < 1 or abs(last_float - float(gt)) / float(gt) < 0.02
            except Exception:
                exact_match = False
        else:
            exact_match = re.search(regex, answer.lower()) is not None
        
        cat_exact_match.add(cat, exact_match)
        
    
    # always check plan

    plan_valid = not (row_dict['flag'] == 'error')
    cat_plan_vr.add(cat, plan_valid)

    if plan_valid:
        has_seg = False
        has_det = False
        for action in info_obj['action_history']:
            if (action['action']['id'] == 'semantic_segmentation'):
                has_seg = True
                break
        for action in info_obj['action_history']:
            if (action['action']['id'] == 'object_detection'):
                has_det = True
                break
        if 'existence' in cat:
            if (not has_seg) and (not has_det):
                cat_plan_r.add(cat, False)
            else:
                cat_plan_r.add(cat, True)
                cat_plan_p.add(cat, True)
        elif cat in ['counting', 'counting_building', 'det']:
            # det only
            if has_det:
                cat_plan_r.add(cat, True)
                cat_plan_p.add(cat, True)
            else:
                cat_plan_r.add(cat, False)
            if has_seg:
                cat_plan_p.add(cat, False)
        else:
            # seg only
            if has_seg:
                cat_plan_r.add(cat, True)
                cat_plan_p.add(cat, True)
            else:
                cat_plan_r.add(cat, False)
            
            if has_det:
                cat_plan_p.add(cat, False)





            
    
    cat_plans.add(cat, evaluation_res['plan'])
    

print('exact_match:')
cat_exact_match.print_stat()
print('GPTScore:')
cat_results.print_stat()

print('planning:')
cat_plans.print_stat()
print('planning VR')
cat_plan_vr.print_stat()
print('planning P')
cat_plan_p.print_stat()
print('planning R')
cat_plan_r.print_stat()

print('easycopy')
for meter in [cat_plan_vr, cat_plan_p, cat_plan_r, cat_exact_match, cat_results]:
    print(f"{meter.get_mean():.4f}, ", end='')
print()

# Close the database connection
conn.close()
