import os
import json
import copy
import bcrypt

LOCAL_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_data.json")

def _init_seed_data():
    hashed_pwd = bcrypt.hashpw("teacher123".encode(), bcrypt.gensalt()).decode()
    return {
        "teachers": [
            {
                "teacher_id": 1,
                "username": "teacher",
                "password": hashed_pwd,
                "name": "Prof. Demo Teacher"
            }
        ],
        "students": [],
        "subjects": [
            {
                "subject_id": 1,
                "subject_code": "CS101",
                "name": "Artificial Intelligence",
                "section": "Sec-A",
                "teacher_id": 1
            }
        ],
        "subject_students": [],
        "attendance_logs": []
    }

class MockResponse:
    def __init__(self, data=None):
        self.data = data if data is not None else []

class MockQueryBuilder:
    def __init__(self, table_name, client):
        self.table_name = table_name
        self.client = client
        self.filters = []  # list of (column, value)
        self.action = "select"
        self.selected_cols = "*"
        self.insert_data = None
        self.update_data = None

    def select(self, cols="*"):
        self.action = "select"
        self.selected_cols = cols
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def insert(self, data):
        self.action = "insert"
        self.insert_data = copy.deepcopy(data)
        return self

    def delete(self):
        self.action = "delete"
        return self

    def update(self, data):
        self.action = "update"
        self.update_data = copy.deepcopy(data)
        return self

    def execute(self):
        db = self.client.load_db()
        table = db.setdefault(self.table_name, [])

        if self.action == "insert":
            inserted_records = []
            records = self.insert_data if isinstance(self.insert_data, list) else [self.insert_data]
            
            id_key = {
                "teachers": "teacher_id",
                "students": "student_id",
                "subjects": "subject_id",
                "subject_students": "id",
                "attendance_logs": "id"
            }.get(self.table_name, "id")

            existing_ids = [r[id_key] for r in table if id_key in r and isinstance(r[id_key], int)]
            next_id = (max(existing_ids) + 1) if existing_ids else 1

            for r in records:
                new_row = copy.deepcopy(r)
                if id_key not in new_row or new_row[id_key] is None:
                    new_row[id_key] = next_id
                    next_id += 1
                table.append(new_row)
                inserted_records.append(new_row)

            self.client.save_db(db)
            return MockResponse(inserted_records if isinstance(self.insert_data, list) else inserted_records[:1])

        elif self.action == "delete":
            matched = []
            remaining = []
            for row in table:
                match = True
                for col, val in self.filters:
                    if str(row.get(col)) != str(val):
                        match = False
                        break
                if match:
                    matched.append(row)
                else:
                    remaining.append(row)
            db[self.table_name] = remaining
            self.client.save_db(db)
            return MockResponse(matched)

        elif self.action == "update":
            updated_records = []
            for row in table:
                match = True
                for col, val in self.filters:
                    if str(row.get(col)) != str(val):
                        match = False
                        break
                if match:
                    row.update(self.update_data)
                    updated_records.append(row)
            self.client.save_db(db)
            return MockResponse(updated_records)

        # Default action: "select"
        result = []
        for row in table:
            match = True
            for col, val in self.filters:
                # Handle special filter syntax like subjects.teacher_id
                if "." in col:
                    parent_col, child_col = col.split(".", 1)
                    if parent_col == "subjects" and self.table_name == "attendance_logs":
                        subj_id = row.get("subject_id")
                        subj = next((s for s in db.get("subjects", []) if s.get("subject_id") == subj_id), {})
                        if str(subj.get(child_col)) != str(val):
                            match = False
                            break
                    else:
                        match = False
                        break
                elif str(row.get(col)) != str(val):
                    match = False
                    break

            if match:
                row_copy = copy.deepcopy(row)

                # Handle relational expansions
                # 1. subjects: subject_students(count), attendance_logs(timestamp)
                if self.table_name == "subjects" and "subject_students" in self.selected_cols:
                    sub_id = row_copy.get("subject_id")
                    enrolled_count = len([ss for ss in db.get("subject_students", []) if ss.get("subject_id") == sub_id])
                    logs = [
                        {"timestamp": al.get("timestamp")}
                        for al in db.get("attendance_logs", [])
                        if al.get("subject_id") == sub_id
                    ]
                    row_copy["subject_students"] = [{"count": enrolled_count}]
                    row_copy["attendance_logs"] = logs

                # 2. subject_students: *, subjects(*)
                if self.table_name == "subject_students" and "subjects(*)" in self.selected_cols:
                    sub_id = row_copy.get("subject_id")
                    subj = next((s for s in db.get("subjects", []) if s.get("subject_id") == sub_id), None)
                    row_copy["subjects"] = subj

                # 3. subject_students: *, students(*)
                if self.table_name == "subject_students" and "students(*)" in self.selected_cols:
                    stud_id = row_copy.get("student_id")
                    student = next((st for st in db.get("students", []) if st.get("student_id") == stud_id), None)
                    row_copy["students"] = student

                # 4. attendance_logs: *, subjects(*) or subjects!inner(*)
                if self.table_name == "attendance_logs" and "subjects" in self.selected_cols:
                    sub_id = row_copy.get("subject_id")
                    subj = next((s for s in db.get("subjects", []) if s.get("subject_id") == sub_id), None)
                    row_copy["subjects"] = subj

                result.append(row_copy)

        return MockResponse(result)

class MockSupabaseClient:
    def __init__(self, file_path=LOCAL_DB_FILE):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            self.save_db(_init_seed_data())

    def load_db(self):
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except Exception:
            data = _init_seed_data()
            self.save_db(data)
            return data

    def save_db(self, data):
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def table(self, table_name):
        return MockQueryBuilder(table_name, self)

_mock_instance = None

def get_mock_client():
    global _mock_instance
    if _mock_instance is None:
        _mock_instance = MockSupabaseClient()
    return _mock_instance
