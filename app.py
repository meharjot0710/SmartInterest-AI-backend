from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib
import numpy as np
import os
import json
import random
from db import users_collection

app = Flask(__name__)
CORS(app)

@app.route("/store_user", methods=["POST"])
def store_user():
    data = request.json
    user_id = data.get("uid")
    email = data.get("email")
    name = data.get("name")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    existing_user = users_collection.find_one({"uid": user_id})
    if not existing_user:
        users_collection.insert_one({
            "uid": user_id,
            "email": email,
            "name": name,
            "scores": {},  
            "projects": [],  
            "predicted_interest": None,
            "roadmap": None,
            "profilePhoto":"https://i.pravatar.cc/150?img=64"
        })
    return jsonify({"message": "User stored successfully!"})

@app.route("/get_user_data", methods=["GET"])
def get_user_data():
    user_id = request.args.get("uid")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    user_data = users_collection.find_one({"uid": user_id}, {"_id": 0})
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user_data)

@app.route("/predict", methods=["POST"])
def predict_interest():
    model_path = os.path.join("model", "smartinterest_model_phase2.pkl")
    model = joblib.load(model_path)
    data = request.get_json()
    input_data = np.array([
        float(data["Operating System"]),
        float(data["DSA"]),
        float(data["Frontend"]),
        float(data["Backend"]),
        float(data["Machine Learning"]),
        float(data["Data Analytics"]),
        int(data["Project 1"]),
        int(data["Level1"]),
        int(data["Project 2"]),
        int(data["Level2"]),
        int(data["Project 3"]),
        int(data["Level3"]),
        int(data["Project 4"]),
        int(data["Level4"]),
    ]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return jsonify({
        "predicted_interest": prediction
    })

@app.route("/store_project", methods=["POST"])
def store_project():
    data = request.json
    user_id = data.get("uid")
    project = data.get("project")  
    if not user_id or not project:
        return jsonify({"error": "User ID and Project are required"}), 400
    users_collection.update_one(
        {"uid": user_id},
        {"$push": {"projects": project}}
    )
    return jsonify({"message": "Project stored successfully!"})

@app.route("/roadmaps", methods=["GET"])
def get_roadmaps():
    roadmap_path = "roadmap_resources.json"
    with open(roadmap_path, "r") as f:
        roadmaps = json.load(f)
    return jsonify(roadmaps)

@app.route("/get_questions", methods=["GET"])
def get_questions():
    with open("questions.json", "r") as f:
        questions_data = json.load(f)
    subject = request.args.get("subject")
    if subject in questions_data:
        questions = questions_data[subject]
        random.shuffle(questions)  
        return jsonify({"questions": questions})
    else:
        return jsonify({"error": "Subject not found"}), 404

@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    with open("questions.json", "r") as f:
        questions_data = json.load(f)
    data = request.get_json()
    subject = data.get("subject")
    user_answers = data.get("answers")  
    if subject not in questions_data:
        return jsonify({"error": "Invalid subject"}), 400
    questions = questions_data[subject]
    correct_answers = {idx: q["answer"] for idx, q in enumerate(questions)}
    score = sum(1 for idx, ans in enumerate(user_answers) if correct_answers.get(idx) == ans)
    total_questions = len(questions)
    percentage = (score / total_questions) * 100 if total_questions else 0
    return jsonify({
        "score": score,
        "total": total_questions,
        "percentage": round(percentage, 2)
    })

@app.route("/update_user_data", methods=["POST"])
def update_user_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    user_id = data.get("uid")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    user = users_collection.find_one({"uid": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    try:
        updated_scores = user.get("scores", {})
        subjects=["Operating System", "DSA", "Frontend", "Backend", "Machine Learning", "Data Analytics"]
        for subject in subjects:
            if subject not in updated_scores:
                updated_scores[subject] = []
            updated_scores[subject].append(float(data['formdata'][subject]*10))
            if len(updated_scores[subject]) > 3:
                updated_scores[subject] = updated_scores[subject][-3:]
        updated_projects = [
            [data['formdata']["Project 1"],data['formdata']["Level1"]],
            [data['formdata']["Project 2"],data['formdata']["Level2"]],
            [data['formdata']["Project 3"],data['formdata']["Level3"]],
            [data['formdata']["Project 4"],data['formdata']["Level4"]],
        ]
        updated_projects = [p for p in updated_projects if p]
        interest_label = data['predicted_interest']
        updated_interests=interest_label
        users_collection.update_one(
            {"uid": user_id},
            {"$set": {
                "scores": updated_scores,
                "projects": updated_projects,
                "predicted_interest": updated_interests,
                "roadmap":data['roadmap']
            }}
        )
        return jsonify({
            "message": "User data updated successfully",
            "latest_interest": interest_label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)