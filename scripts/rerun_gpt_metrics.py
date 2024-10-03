import sqlite3
import json
import sys
import tqdm
sys.path.append('.')
import llm_utils
import llm_metrics

def rerun_func(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
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

    for row in tqdm.tqdm(rows):
        # Create a dictionary for each row
        row_dict = dict(zip(column_names, row))

        # Load the 'info' field as a JSON object
        info_obj = json.loads(row_dict['info'])
        
        # Extract the values needed for computation
        label_type = info_obj.get('label_type', None)
        q = row_dict['question']
        gt = row_dict['answer_gt']
        answer = row_dict['answer']
        
        ans_correct = llm_utils.retry_until_succeed(
            lambda: llm_metrics.compare_question_answer_groundtruth(q, answer, gt, need_confirm=False)
        )
        info_obj['new_ans_correct'] = ans_correct

        # Convert the modified info_obj back to a JSON string
        updated_info_json = json.dumps(info_obj)
        
        # Update the 'info' field in the database
        update_query = f'''
        UPDATE {table_name}
        SET info = ?
        WHERE id = ?
        '''
        
        # Execute the update query
        cursor.execute(update_query, (updated_info_json, row_dict['id']))

        # Commit the changes to the database
        conn.commit()

    # Close the connection
    conn.close()

import logging
llm_utils.setup_root_logger(
    console=False,
    filename='llm_metrics.log', 
    level=logging.INFO)
for db_path in [
    'with_det_1k_real_tunedglm.db',
    'with_det_1k_real.db',
    'with_det_1k_real_tier2.db'
    'with_det_1k_real_qwen2.5.db',
    'with_det_1k_real_geochat.db',
    'with_det_1k_real_dpsk.db'
]:
    print(">>>>>>>>>>>>>>>>>> running db: ", db_path)
    rerun_func(db_path)
