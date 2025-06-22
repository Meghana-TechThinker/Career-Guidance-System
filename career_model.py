import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import joblib
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class CareerRecommender:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.subject_binarizer = None
        self.career_area_binarizer = None
        self.company_type_binarizer = None
        self.career_paths = [
            "Software Developer",
            "Data Scientist",
            "Business Analyst",
            "UX Designer",
            "Project Manager",
            "DevOps Engineer",
            "Cloud Architect",
            "Cybersecurity Analyst",
            "Mobile Developer",
            "Game Developer",
            "Blockchain Developer",
            "Robotics Engineer",
            "AI/ML Engineer",
            "Full Stack Developer",
            "QA Engineer",
            "System Administrator",
            "Network Engineer",
            "Database Administrator",
            "Technical Lead",
            "Product Manager"
        ]
        
        self.career_attributes = {
            "Software Developer": {
                "programming_concept_percentage": (70, 100),
                "communication_skills_percentage": (50, 80),
                "hours_working_per_day": (6, 10),
                "hackathons": (1, 5),
                "coding_skills_rating": (8, 10),
                "public_speaking_points": (4, 8),
                "self_learning_capability": (7, 10),
                "certifications": (1, 3),
                "workshops": (2, 5),
                "interested_subjects": ["Web Development", "Mobile Development", "Game Development"],
                "interested_career_area": ["Technology"],
                "company_type": ["Startup", "Product", "Service"]
            },
            "Data Scientist": {
                "programming_concept_percentage": (60, 90),
                "communication_skills_percentage": (60, 90),
                "hours_working_per_day": (6, 9),
                "hackathons": (1, 4),
                "coding_skills_rating": (7, 10),
                "public_speaking_points": (5, 9),
                "self_learning_capability": (8, 10),
                "certifications": (2, 4),
                "workshops": (3, 6),
                "interested_subjects": ["Data Science", "AI/ML", "Cloud Computing"],
                "interested_career_area": ["Technology", "Research"],
                "company_type": ["MNC", "Product", "Service"]
            },
            "DevOps Engineer": {
                "programming_concept_percentage": (65, 95),
                "communication_skills_percentage": (55, 85),
                "hours_working_per_day": (7, 10),
                "hackathons": (1, 3),
                "coding_skills_rating": (7, 9),
                "public_speaking_points": (4, 8),
                "self_learning_capability": (8, 10),
                "certifications": (2, 5),
                "workshops": (2, 5),
                "interested_subjects": ["Cloud Computing", "DevOps", "Cybersecurity"],
                "interested_career_area": ["Technology"],
                "company_type": ["MNC", "Product", "Service"]
            },
            "Cybersecurity Analyst": {
                "programming_concept_percentage": (60, 90),
                "communication_skills_percentage": (50, 80),
                "hours_working_per_day": (6, 9),
                "hackathons": (2, 5),
                "coding_skills_rating": (6, 9),
                "public_speaking_points": (5, 9),
                "self_learning_capability": (8, 10),
                "certifications": (2, 4),
                "workshops": (3, 6),
                "interested_subjects": ["Cybersecurity", "Cloud Computing", "Blockchain"],
                "interested_career_area": ["Technology"],
                "company_type": ["MNC", "Service"]
            },
            "Mobile Developer": {
                "programming_concept_percentage": (70, 100),
                "communication_skills_percentage": (50, 80),
                "hours_working_per_day": (6, 10),
                "hackathons": (1, 4),
                "coding_skills_rating": (8, 10),
                "public_speaking_points": (4, 8),
                "self_learning_capability": (7, 10),
                "certifications": (1, 3),
                "workshops": (2, 5),
                "interested_subjects": ["Mobile Development", "Web Development", "Game Development"],
                "interested_career_area": ["Technology"],
                "company_type": ["Startup", "Product", "Service"]
            },
            "Game Developer": {
                "programming_concept_percentage": (75, 100),
                "communication_skills_percentage": (50, 80),
                "hours_working_per_day": (7, 11),
                "hackathons": (1, 4),
                "coding_skills_rating": (8, 10),
                "public_speaking_points": (4, 8),
                "self_learning_capability": (7, 10),
                "certifications": (1, 3),
                "workshops": (2, 5),
                "interested_subjects": ["Game Development", "Mobile Development", "Web Development"],
                "interested_career_area": ["Technology", "Creative"],
                "company_type": ["Startup", "Product"]
            },
            "Blockchain Developer": {
                "programming_concept_percentage": (70, 100),
                "communication_skills_percentage": (50, 80),
                "hours_working_per_day": (6, 10),
                "hackathons": (1, 4),
                "coding_skills_rating": (8, 10),
                "public_speaking_points": (4, 8),
                "self_learning_capability": (8, 10),
                "certifications": (2, 4),
                "workshops": (2, 5),
                "interested_subjects": ["Blockchain", "Web Development", "Cybersecurity"],
                "interested_career_area": ["Technology"],
                "company_type": ["Startup", "Product"]
            },
            "Robotics Engineer": {
                "programming_concept_percentage": (65, 95),
                "communication_skills_percentage": (55, 85),
                "hours_working_per_day": (6, 9),
                "hackathons": (1, 4),
                "coding_skills_rating": (7, 9),
                "public_speaking_points": (5, 9),
                "self_learning_capability": (8, 10),
                "certifications": (2, 4),
                "workshops": (3, 6),
                "interested_subjects": ["Robotics", "AI/ML", "Mobile Development"],
                "interested_career_area": ["Technology", "Research"],
                "company_type": ["MNC", "Product"]
            },
            "AI/ML Engineer": {
                "programming_concept_percentage": (65, 95),
                "communication_skills_percentage": (55, 85),
                "hours_working_per_day": (6, 9),
                "hackathons": (1, 4),
                "coding_skills_rating": (7, 9),
                "public_speaking_points": (5, 9),
                "self_learning_capability": (8, 10),
                "certifications": (2, 4),
                "workshops": (3, 6),
                "interested_subjects": ["AI/ML", "Data Science", "Robotics"],
                "interested_career_area": ["Technology", "Research"],
                "company_type": ["MNC", "Product"]
            },
            "Full Stack Developer": {
                "programming_concept_percentage": (75, 100),
                "communication_skills_percentage": (50, 80),
                "hours_working_per_day": (6, 10),
                "hackathons": (1, 4),
                "coding_skills_rating": (8, 10),
                "public_speaking_points": (4, 8),
                "self_learning_capability": (7, 10),
                "certifications": (1, 3),
                "workshops": (2, 5),
                "interested_subjects": ["Web Development", "Mobile Development", "Cloud Computing"],
                "interested_career_area": ["Technology"],
                "company_type": ["Startup", "Product", "Service"]
            }
        }

        # Initialize the model and transformers
        self.load_or_create_model()

    def load_or_create_model(self):
        """Load existing model and transformers or create new ones."""
        try:
            # Check if model and transformers exist
            if (os.path.exists('career_model.joblib') and 
                os.path.exists('career_scaler.joblib') and
                os.path.exists('career_subject_binarizer.joblib') and
                os.path.exists('career_career_area_binarizer.joblib') and
                os.path.exists('career_company_type_binarizer.joblib')):
                
                # Load existing model and transformers
                self.model = joblib.load('career_model.joblib')
                self.scaler = joblib.load('career_scaler.joblib')
                self.subject_binarizer = joblib.load('career_subject_binarizer.joblib')
                self.career_area_binarizer = joblib.load('career_career_area_binarizer.joblib')
                self.company_type_binarizer = joblib.load('career_company_type_binarizer.joblib')
                print("Loaded existing model and transformers")
            else:
                # Initialize new transformers
                self.scaler = StandardScaler()
                self.subject_binarizer = MultiLabelBinarizer()
                self.career_area_binarizer = MultiLabelBinarizer()
                self.company_type_binarizer = MultiLabelBinarizer()
                
                # Create and train new model
                self.create_sample_dataset()
                self.train_model()
                print("Created and trained new model")
        except Exception as e:
            print(f"Error in load_or_create_model: {str(e)}")
            # Initialize new transformers as fallback
            self.scaler = StandardScaler()
            self.subject_binarizer = MultiLabelBinarizer()
            self.career_area_binarizer = MultiLabelBinarizer()
            self.company_type_binarizer = MultiLabelBinarizer()
            self.create_sample_dataset()
            self.train_model()

    def create_sample_dataset(self):
        """Create a sample dataset for training the career recommendation model."""
        print("Generating sample dataset...")
        data = []
        
        # Generate samples for each career path
        for career, attributes in self.career_attributes.items():
            print(f"Generating samples for {career}...")
            for _ in range(100):  # Generate 100 samples per career
                sample = {
                    'career': career,
                    'programming_concept_percentage': np.random.randint(*attributes['programming_concept_percentage']),
                    'communication_skills_percentage': np.random.randint(*attributes['communication_skills_percentage']),
                    'hours_working_per_day': np.random.randint(*attributes['hours_working_per_day']),
                    'hackathons': np.random.randint(*attributes['hackathons']),
                    'coding_skills_rating': np.random.randint(*attributes['coding_skills_rating']),
                    'public_speaking_points': np.random.randint(*attributes['public_speaking_points']),
                    'self_learning_capability': np.random.randint(*attributes['self_learning_capability']),
                    'certifications': np.random.randint(*attributes['certifications']),
                    'workshops': np.random.randint(*attributes['workshops']),
                    'interested_subjects': np.random.choice(attributes['interested_subjects'], 
                                                          size=np.random.randint(1, len(attributes['interested_subjects'])+1), 
                                                          replace=False).tolist(),
                    'interested_career_area': np.random.choice(attributes['interested_career_area'], 
                                                             size=np.random.randint(1, len(attributes['interested_career_area'])+1), 
                                                             replace=False).tolist(),
                    'company_type': np.random.choice(attributes['company_type'])
                }
                data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save the dataset
        print("Saving dataset to career_dataset.csv...")
        df.to_csv('career_dataset.csv', index=False)
        
        # Save career attributes
        print("Saving career attributes to career_attributes.json...")
        with open('career_attributes.json', 'w') as f:
            json.dump(self.career_attributes, f, indent=4)
        
        print("Dataset creation completed successfully.")
        return df

    def prepare_training_data(self, df, numerical_features, categorical_features):
        """Prepare training data by scaling numerical features and encoding categorical features."""
        try:
            # Scale numerical features
            numerical_data = df[numerical_features].values
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(numerical_data)
            scaled_numerical = self.scaler.transform(numerical_data)
            
            # Encode categorical features
            categorical_data = []
            for feature, binarizer in categorical_features.items():
                # Convert string lists to actual lists if needed
                feature_data = df[feature].apply(lambda x: x if isinstance(x, list) else [x])
                
                # Fit and transform the binarizer
                if not hasattr(binarizer, 'classes_'):
                    binarizer.fit(feature_data)
                encoded_data = binarizer.transform(feature_data)
                categorical_data.append(encoded_data)
            
            # Combine all features
            if categorical_data:
                all_features = np.hstack([scaled_numerical] + categorical_data)
            else:
                all_features = scaled_numerical
                
            return all_features
            
        except Exception as e:
            print(f"Error in prepare_training_data: {str(e)}")
            raise

    def train_model(self):
        """Train the career recommendation model with improved accuracy."""
        try:
            # Create dataset if it doesn't exist
            if not os.path.exists('career_dataset.csv'):
                print("Creating sample dataset...")
                self.create_sample_dataset()
            
            # Load dataset
            print("Loading dataset...")
            df = pd.read_csv('career_dataset.csv')
            
            # Check if dataset has the required columns
            required_columns = ['career', 'programming_concept_percentage', 'communication_skills_percentage', 
                               'coding_skills_rating', 'public_speaking_points', 'self_learning_capability',
                               'hours_working_per_day', 'hackathons', 'certifications', 'workshops',
                               'interested_subjects', 'interested_career_area', 'company_type']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns in dataset: {missing_columns}")
                print("Recreating dataset...")
                self.create_sample_dataset()
                df = pd.read_csv('career_dataset.csv')
            
            # Prepare features
            numerical_features = [
                'programming_concept_percentage',
                'communication_skills_percentage',
                'coding_skills_rating',
                'public_speaking_points',
                'self_learning_capability',
                'hours_working_per_day',
                'hackathons',
                'certifications',
                'workshops'
            ]
            
            categorical_features = {
                'interested_subjects': self.subject_binarizer,
                'interested_career_area': self.career_area_binarizer,
                'company_type': self.company_type_binarizer
            }
            
            # Prepare training data
            X = self.prepare_training_data(df, numerical_features, categorical_features)
            y = df['career']
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Create an ensemble of models
            from sklearn.ensemble import VotingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            # Initialize base models
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
            svc = SVC(probability=True, random_state=42)
            
            # Create voting classifier
            self.model = VotingClassifier(
                estimators=[
                    ('rf', rf),
                    ('lr', lr),
                    ('svc', svc)
                ],
                voting='soft'
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Calculate detailed metrics
            from sklearn.metrics import classification_report, confusion_matrix
            y_pred = self.model.predict(X_test)
            print("\nModel Performance Metrics:")
            print(classification_report(y_test, y_pred))
            
            print(f"\nModel training score: {train_score:.2f}")
            print(f"Model test score: {test_score:.2f}")
            
            # Save model and transformers
            joblib.dump(self.model, 'career_model.joblib')
            joblib.dump(self.scaler, 'career_scaler.joblib')
            joblib.dump(self.subject_binarizer, 'career_subject_binarizer.joblib')
            joblib.dump(self.career_area_binarizer, 'career_career_area_binarizer.joblib')
            joblib.dump(self.company_type_binarizer, 'career_company_type_binarizer.joblib')
            
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            raise

    def get_recommendations(self, user_data):
        """Get career recommendations with improved confidence calculation."""
        try:
            # Prepare numerical features
            numerical_features = [
                user_data['skills']['programming_concept_percentage'],
                user_data['skills']['communication_skills_percentage'],
                user_data['skills']['coding_skills_rating'],
                user_data['skills']['public_speaking_points'],
                user_data['skills']['self_learning_capability'],
                user_data['experience']['hours_working_per_day'],
                user_data['experience']['hackathons'],
                user_data['experience']['certifications'],
                user_data['experience']['workshops']
            ]
            numerical_data = np.array([numerical_features])
            scaled_numerical = self.scaler.transform(numerical_data)
            
            # Prepare categorical features
            categorical_data = []
            
            # Transform subjects
            subjects = user_data['interests']['interested_subjects']
            subjects_encoded = self.subject_binarizer.transform([subjects])
            categorical_data.append(subjects_encoded)
            
            # Transform career areas
            career_areas = user_data['interests']['interested_career_area']
            career_areas_encoded = self.career_area_binarizer.transform([career_areas])
            categorical_data.append(career_areas_encoded)
            
            # Transform company type
            company_type = [[user_data['interests']['company_type']]]
            company_type_encoded = self.company_type_binarizer.transform(company_type)
            categorical_data.append(company_type_encoded)
            
            # Combine all features
            X = np.hstack([scaled_numerical] + categorical_data)
            
            # Get predictions and probabilities
            predictions = self.model.predict_proba(X)[0]
            
            # Get top 3 recommendations with confidence threshold
            confidence_threshold = 0.15  # Minimum confidence threshold
            top_indices = np.argsort(predictions)[::-1]
            recommendations = []
            
            for idx in top_indices:
                career = self.model.classes_[idx]
                confidence = predictions[idx]
                
                # Only include recommendations above confidence threshold
                if confidence >= confidence_threshold and len(recommendations) < 3:
                    # Get career details from attributes
                    career_details = self.career_attributes.get(career, {})
                    
                    # Calculate match score based on attribute alignment
                    match_score = self._calculate_match_score(user_data, career_details)
                    
                    # Adjust confidence based on match score
                    adjusted_confidence = (confidence * 0.7 + match_score * 0.3) * 100
                    
                    recommendations.append({
                        'career': career,
                        'confidence': round(adjusted_confidence, 2),
                        'description': career_details.get('description', ''),
                        'required_skills': career_details.get('required_skills', []),
                        'growth_potential': career_details.get('growth_potential', ''),
                        'salary_range': career_details.get('salary_range', ''),
                        'job_roles': career_details.get('job_roles', []),
                        'match_details': {
                            'skills_match': match_score * 100,
                            'interests_match': self._calculate_interests_match(user_data['interests'], career_details),
                            'experience_match': self._calculate_experience_match(user_data['experience'], career_details)
                        }
                    })
            
            return recommendations
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            raise

    def _calculate_match_score(self, user_data, career_details):
        """Calculate a detailed match score between user profile and career requirements."""
        try:
            # Skills match (40% weight)
            skills_score = 0
            if 'programming_concept_percentage' in career_details:
                user_score = user_data['skills']['programming_concept_percentage']
                min_req, max_req = career_details['programming_concept_percentage']
                skills_score += min(1.0, (user_score - min_req) / (max_req - min_req))
            
            # Experience match (30% weight)
            experience_score = 0
            if 'hackathons' in career_details:
                user_hackathons = user_data['experience']['hackathons']
                min_req, max_req = career_details['hackathons']
                experience_score += min(1.0, (user_hackathons - min_req) / (max_req - min_req))
            
            # Interests match (30% weight)
            interests_score = self._calculate_interests_match(user_data['interests'], career_details)
            
            # Calculate weighted average
            match_score = (skills_score * 0.4 + experience_score * 0.3 + interests_score * 0.3)
            return min(1.0, max(0.0, match_score))
            
        except Exception as e:
            print(f"Error in _calculate_match_score: {str(e)}")
            return 0.0

    def _calculate_interests_match(self, user_interests, career_details):
        """Calculate how well user interests match career requirements."""
        try:
            match_count = 0
            total_subjects = len(career_details.get('interested_subjects', []))
            
            if total_subjects == 0:
                return 0.0
            
            for subject in user_interests['interested_subjects']:
                if subject in career_details.get('interested_subjects', []):
                    match_count += 1
            
            return match_count / total_subjects
            
        except Exception as e:
            print(f"Error in _calculate_interests_match: {str(e)}")
            return 0.0

    def _calculate_experience_match(self, user_experience, career_details):
        """Calculate how well user experience matches career requirements."""
        try:
            match_score = 0
            total_metrics = 3  # Number of experience metrics we consider
            
            # Check hours working per day
            if 'hours_working_per_day' in career_details:
                user_hours = user_experience['hours_working_per_day']
                min_req, max_req = career_details['hours_working_per_day']
                if min_req <= user_hours <= max_req:
                    match_score += 1
            
            # Check certifications
            if 'certifications' in career_details:
                user_certs = user_experience['certifications']
                min_req, max_req = career_details['certifications']
                if min_req <= user_certs <= max_req:
                    match_score += 1
            
            # Check workshops
            if 'workshops' in career_details:
                user_workshops = user_experience['workshops']
                min_req, max_req = career_details['workshops']
                if min_req <= user_workshops <= max_req:
                    match_score += 1
            
            return match_score / total_metrics
            
        except Exception as e:
            print(f"Error in _calculate_experience_match: {str(e)}")
            return 0.0

    def _generate_explanation(self, career, skills, experience, interests, education, personality):
        """Generate a personalized explanation for why this career was recommended."""
        explanation = f"Based on your profile, {career} appears to be a good match. "
        
        # Add skill-based explanation
        if skills.get('programming_concept_percentage', 0) >= 70:
            explanation += "Your strong programming skills align well with this role. "
        elif skills.get('programming_concept_percentage', 0) >= 50:
            explanation += "Your programming skills are adequate for this role. "
        
        # Add experience-based explanation
        if experience.get('hackathons', 0) > 0:
            explanation += f"Your participation in {experience.get('hackathons')} hackathons demonstrates practical experience. "
        
        # Add interest-based explanation
        if interests.get('interested_subjects') and any(subject in self.career_attributes[career]['interested_subjects'] for subject in interests.get('interested_subjects', [])):
            explanation += "Your interests align with the subjects relevant to this career. "
        
        # Add education-based explanation
        if education:
            explanation += f"Your {education} background provides a good foundation for this career path. "
        
        # Add personality-based explanation
        if personality:
            explanation += f"Your personality traits ({', '.join(personality)}) are well-suited for this role. "
        
        return explanation 