from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
from career_model import CareerRecommender
import joblib
from urllib.parse import unquote
from career_roadmaps import CAREER_ROADMAPS

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Initialize the career recommender
career_recommender = CareerRecommender()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/assessment')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        # Extract user inputs from the structured data
        user_data = {
            'skills': {
                'programming_concept_percentage': int(data['skills']['programming_concept_percentage']),
                'communication_skills_percentage': int(data['skills']['communication_skills_percentage']),
                'coding_skills_rating': int(data['skills']['coding_skills_rating']),
                'public_speaking_points': int(data['skills']['public_speaking_points']),
                'self_learning_capability': int(data['skills']['self_learning_capability'])
            },
            'experience': {
                'hours_working_per_day': int(data['experience']['hours_working_per_day']),
                'hackathons': int(data['experience']['hackathons']),
                'certifications': int(data['experience']['certifications']),
                'workshops': int(data['experience']['workshops'])
            },
            'interests': {
                'interested_subjects': data['interests']['interested_subjects'],
                'interested_career_area': data['interests']['interested_career_area'],
                'company_type': data['interests']['company_type']
            }
        }
        
        print("Processed user_data:", user_data)  # Debug print
        
        # Get recommendations from the model
        recommendations = career_recommender.get_recommendations(user_data)
        
        # Add descriptions and required skills for each recommendation
        for rec in recommendations:
            career = rec['career']
            # Check if career exists in roadmaps
            if career in CAREER_ROADMAPS:
                rec['has_roadmap'] = True
                foundation_skills = CAREER_ROADMAPS[career]['foundation'][:3]  # Get first 3 foundation skills
                rec['description'] = f"A {career} role focuses on {', '.join(foundation_skills).lower()} and other key skills."
                rec['required_skills'] = [
                    "Technical expertise in relevant tools and technologies",
                    "Problem-solving and analytical thinking",
                    "Communication and collaboration skills"
                ]
                rec['growth_opportunities'] = [
                    "Career advancement to senior roles",
                    "Opportunity to work on innovative projects",
                    "Continuous learning and skill development"
                ]
            else:
                rec['has_roadmap'] = False
                rec['description'] = f"A {career} typically works with cutting-edge technology and requires strong technical skills."
                rec['required_skills'] = [
                    "Technical expertise in the field",
                    "Problem-solving abilities",
                    "Team collaboration"
                ]
                rec['growth_opportunities'] = [
                    "Career growth opportunities",
                    "Industry recognition",
                    "Professional development"
                ]
        
        return jsonify(recommendations)
        
    except Exception as e:
        print("Error in recommend route:", str(e))  # Debug print
        return jsonify({
            'error': 'Failed to get career recommendations',
            'details': str(e)
        }), 400

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/roadmap/<career>')
def career_roadmap(career):
    # URL decode the career parameter
    career = unquote(career)
    
    # Get the roadmap data for the career
    roadmap = CAREER_ROADMAPS.get(career, {
        'foundation': [
            'Learn fundamental concepts',
            'Build basic skills',
            'Understand industry standards'
        ],
        'intermediate': [
            'Gain practical experience',
            'Develop specialized skills',
            'Work on real projects'
        ],
        'advanced': [
            'Master advanced concepts',
            'Lead projects',
            'Contribute to the community'
        ],
        'expert': [
            'Innovate in the field',
            'Mentor others',
            'Shape industry trends'
        ]
    })
    
    return render_template('roadmap.html', career=career, roadmap=roadmap)

if __name__ == '__main__':
    app.run(debug=True) 