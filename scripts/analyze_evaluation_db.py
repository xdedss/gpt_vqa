
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import database_sqlite

def analyze_flag(db_path):
    db = database_sqlite.Database(db_path)
    errors = db.count_data_with_flag('error')
    good = db.count_data_with_flag('correct')
    bad = db.count_data_with_flag('incorrect')
    total = errors + good + bad
    print(f'errors: {errors}, good: {good}, bad: {bad}')
    print(f'errors: {errors/total:.4f}, good: {good/total:.4f}, bad: {bad/total:.4f}')

analyze_flag('simple_agent_bsb_feedback.db')

