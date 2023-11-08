import sqlite3

class Database:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.create_table()

    def create_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS data (
            id INTEGER PRIMARY KEY,
            image_id INTEGER,
            question TEXT,
            answer TEXT,
            answer_gt TEXT,
            flag TEXT,
            info TEXT
        )
        """
        self.conn.execute(sql)
        self.conn.commit()

    def add_data(self, image_id, question, answer, answer_gt, flag, info):
        sql = """
        INSERT INTO data (image_id, question, answer, answer_gt, flag, info)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        data = (image_id, question, answer, answer_gt, flag, info)
        self.conn.execute(sql, data)
        self.conn.commit()
    
    def count_data_with_flag(self, flag):
        sql = """
        SELECT COUNT(*) FROM data WHERE flag = ?
        """
        cursor = self.conn.execute(sql, (flag,))
        count = cursor.fetchone()[0]
        return count

    def close(self):
        self.conn.close()

# Usage example:
if __name__ == "__main__":
    # db = Database("my_database.db")
    # db.add_data(1, "What is this?", "It's a cat.", "It's a cat.", "correct", "Additional info")
    # db.add_data(2, "What color is it?", "It's black.", "It's black.", "correct", "More info")
    # db.close()
    pass
