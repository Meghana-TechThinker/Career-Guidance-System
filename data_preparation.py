import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, StandardScaler
import json

def create_sample_dataset():
    """
    Create a sample dataset for career guidance with the specified attributes.
    """
    # Define the dataset structure
    data = {
        'programming_concept_percentage': [],
        'programming_concepts_percentage': [],
        'communication_skills_percentage': [],
        'hours_working_per_day': [],
        'hackathons': [],
        'coding_skills_rating': [],
        'public_speaking_points': [],
        'self_learning_capability': [],
        'certifications': [],
        'workshops': [],
        'interested_subjects': [],
        'interested_career_area': [],
        'company_type': [],
        'career_path': []  # Target variable
    }
    
    # Career paths and their associated attributes
    career_attributes = {
        "Software Developer": {
            "programming_concept_range": (80, 100),
            "programming_concepts_range": (80, 100),
            "communication_skills_range": (60, 90),
            "hours_working_range": (6, 10),
            "hackathons_range": (2, 10),
            "coding_skills_range": (4, 5),
            "public_speaking_range": (3, 5),
            "self_learning_range": (4, 5),
            "certifications_range": (1, 5),
            "workshops_range": (2, 8),
            "interested_subjects": ["Computer Science", "Programming", "Algorithms", "Data Structures"],
            "interested_career_area": ["Technology", "Software Development", "Web Development"],
            "company_type": ["Tech Startup", "Software Company", "IT Company"]
        },
        "Data Scientist": {
            "programming_concept_range": (70, 100),
            "programming_concepts_range": (70, 100),
            "communication_skills_range": (70, 90),
            "hours_working_range": (6, 9),
            "hackathons_range": (1, 8),
            "coding_skills_range": (4, 5),
            "public_speaking_range": (3, 5),
            "self_learning_range": (4, 5),
            "certifications_range": (2, 6),
            "workshops_range": (3, 10),
            "interested_subjects": ["Statistics", "Machine Learning", "Data Analysis", "Python"],
            "interested_career_area": ["Data Science", "Analytics", "Machine Learning"],
            "company_type": ["Tech Company", "Research Institute", "Data Analytics Company"]
        },
        "Business Analyst": {
            "programming_concept_range": (50, 80),
            "programming_concepts_range": (50, 80),
            "communication_skills_range": (80, 100),
            "hours_working_range": (7, 9),
            "hackathons_range": (0, 3),
            "coding_skills_range": (2, 4),
            "public_speaking_range": (4, 5),
            "self_learning_range": (3, 5),
            "certifications_range": (1, 4),
            "workshops_range": (2, 6),
            "interested_subjects": ["Business Analysis", "Project Management", "Data Analysis"],
            "interested_career_area": ["Business Analysis", "Consulting", "Project Management"],
            "company_type": ["Consulting Firm", "Corporate Company", "Business Solutions"]
        },
        "UX Designer": {
            "programming_concept_range": (40, 70),
            "programming_concepts_range": (40, 70),
            "communication_skills_range": (80, 100),
            "hours_working_range": (6, 9),
            "hackathons_range": (0, 4),
            "coding_skills_range": (2, 4),
            "public_speaking_range": (4, 5),
            "self_learning_range": (4, 5),
            "certifications_range": (1, 4),
            "workshops_range": (3, 8),
            "interested_subjects": ["User Experience", "Design", "Psychology", "Human-Computer Interaction"],
            "interested_career_area": ["Design", "User Experience", "Creative Technology"],
            "company_type": ["Design Agency", "Tech Company", "Creative Studio"]
        },
        "Project Manager": {
            "programming_concept_range": (40, 70),
            "programming_concepts_range": (40, 70),
            "communication_skills_range": (90, 100),
            "hours_working_range": (8, 10),
            "hackathons_range": (0, 2),
            "coding_skills_range": (1, 3),
            "public_speaking_range": (4, 5),
            "self_learning_range": (3, 5),
            "certifications_range": (2, 5),
            "workshops_range": (3, 7),
            "interested_subjects": ["Project Management", "Leadership", "Business Strategy"],
            "interested_career_area": ["Project Management", "Leadership", "Business"],
            "company_type": ["Corporate Company", "Consulting Firm", "Tech Company"]
        }
    }
    
    # Generate multiple samples for each career
    for career, attributes in career_attributes.items():
        for _ in range(100):  # Generate 100 samples per career
            # Generate numerical attributes
            data['programming_concept_percentage'].append(
                np.random.randint(*attributes['programming_concept_range'])
            )
            data['programming_concepts_percentage'].append(
                np.random.randint(*attributes['programming_concepts_range'])
            )
            data['communication_skills_percentage'].append(
                np.random.randint(*attributes['communication_skills_range'])
            )
            data['hours_working_per_day'].append(
                np.random.randint(*attributes['hours_working_range'])
            )
            data['hackathons'].append(
                np.random.randint(*attributes['hackathons_range'])
            )
            data['coding_skills_rating'].append(
                np.random.randint(*attributes['coding_skills_range'])
            )
            data['public_speaking_points'].append(
                np.random.randint(*attributes['public_speaking_range'])
            )
            data['self_learning_capability'].append(
                np.random.randint(*attributes['self_learning_range'])
            )
            data['certifications'].append(
                np.random.randint(*attributes['certifications_range'])
            )
            data['workshops'].append(
                np.random.randint(*attributes['workshops_range'])
            )
            
            # Generate categorical attributes
            num_subjects = np.random.randint(1, len(attributes['interested_subjects']) + 1)
            data['interested_subjects'].append(
                np.random.choice(attributes['interested_subjects'], size=num_subjects, replace=False).tolist()
            )
            
            num_career_areas = np.random.randint(1, len(attributes['interested_career_area']) + 1)
            data['interested_career_area'].append(
                np.random.choice(attributes['interested_career_area'], size=num_career_areas, replace=False).tolist()
            )
            
            data['company_type'].append(
                np.random.choice(attributes['company_type'])
            )
            
            data['career_path'].append(career)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('career_dataset.csv', index=False)
    
    # Save career attributes for reference
    with open('career_attributes.json', 'w') as f:
        json.dump(career_attributes, f, indent=4)
    
    return df

def prepare_training_data(df):
    """
    Prepare the training data by encoding categorical variables and scaling numerical features.
    """
    # Initialize binarizers for categorical variables
    subjects_binarizer = MultiLabelBinarizer()
    career_area_binarizer = MultiLabelBinarizer()
    company_type_binarizer = LabelBinarizer()
    
    # Transform categorical variables
    X_subjects = subjects_binarizer.fit_transform(df['interested_subjects'])
    X_career_area = career_area_binarizer.fit_transform(df['interested_career_area'])
    X_company_type = company_type_binarizer.fit_transform(df['company_type'])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = [
        'programming_concept_percentage',
        'programming_concepts_percentage',
        'communication_skills_percentage',
        'hours_working_per_day',
        'hackathons',
        'coding_skills_rating',
        'public_speaking_points',
        'self_learning_capability',
        'certifications',
        'workshops'
    ]
    X_numerical = scaler.fit_transform(df[numerical_features])
    
    # Combine all features
    X = np.hstack([
        X_numerical,
        X_subjects,
        X_career_area,
        X_company_type
    ])
    
    y = df['career_path'].values
    
    return X, y, subjects_binarizer, career_area_binarizer, company_type_binarizer, scaler

if __name__ == '__main__':
    # Create and save the dataset
    df = create_sample_dataset()
    print("Dataset created and saved to 'career_dataset.csv'")
    
    # Prepare training data
    X, y, subjects_binarizer, career_area_binarizer, company_type_binarizer, scaler = prepare_training_data(df)
    print(f"Training data shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of unique careers: {len(np.unique(y))}") 