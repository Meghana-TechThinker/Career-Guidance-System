"""
Career roadmaps for various technology roles.
Each roadmap includes foundation, intermediate, advanced, and expert levels
with specific skills, tools, and milestones.
"""

CAREER_ROADMAPS = {
    'Software Developer': {
        'foundation': [
            'Core programming concepts (variables, loops, functions)',
            'Object-oriented programming fundamentals',
            'Data structures and algorithms basics',
            'Version control with Git',
            'HTML, CSS, and JavaScript basics',
            'Database fundamentals (SQL)',
            'Command line and development environment setup'
        ],
        'intermediate': [
            'Web frameworks (React, Angular, or Vue)',
            'Backend development (Node.js, Python, Java)',
            'RESTful API design and implementation',
            'Database design and optimization',
            'Testing methodologies (unit, integration)',
            'CI/CD basics',
            'Code review and documentation'
        ],
        'advanced': [
            'System design and architecture',
            'Microservices architecture',
            'Cloud platforms (AWS, Azure, GCP)',
            'Security best practices',
            'Performance optimization',
            'Design patterns and clean code',
            'Team leadership and mentoring'
        ],
        'expert': [
            'Enterprise architecture',
            'Technical strategy and roadmap planning',
            'High-scale system design',
            'Technology innovation and R&D',
            'Technical team management',
            'Software development methodologies',
            'Industry thought leadership'
        ]
    },
    'Data Scientist': {
        'foundation': [
            'Programming in Python/R',
            'Statistics and probability',
            'Linear algebra and calculus',
            'Data structures and algorithms',
            'SQL and database concepts',
            'Data visualization (Matplotlib, Seaborn)',
            'Jupyter notebooks and data analysis tools'
        ],
        'intermediate': [
            'Machine learning fundamentals',
            'Scikit-learn and TensorFlow',
            'Feature engineering techniques',
            'Data preprocessing and cleaning',
            'Exploratory data analysis',
            'Statistical modeling',
            'Deep learning basics'
        ],
        'advanced': [
            'Advanced ML algorithms',
            'Natural language processing',
            'Computer vision',
            'Big data technologies (Spark)',
            'MLOps and model deployment',
            'Time series analysis',
            'A/B testing and experimentation'
        ],
        'expert': [
            'Research and publication',
            'Advanced deep learning',
            'AI ethics and governance',
            'Data strategy leadership',
            'Business analytics',
            'Team management',
            'Industry consulting'
        ]
    },
    'DevOps Engineer': {
        'foundation': [
            'Linux system administration',
            'Scripting (Bash, Python)',
            'Version control (Git)',
            'Basic networking concepts',
            'CI/CD fundamentals',
            'Container basics (Docker)',
            'Cloud fundamentals (AWS/Azure/GCP)'
        ],
        'intermediate': [
            'Container orchestration (Kubernetes)',
            'Infrastructure as Code (Terraform)',
            'Configuration management (Ansible)',
            'Monitoring and logging',
            'Cloud architecture patterns',
            'Security practices',
            'Automated testing'
        ],
        'advanced': [
            'Multi-cloud architecture',
            'Service mesh (Istio)',
            'Advanced Kubernetes',
            'Site reliability engineering',
            'Performance optimization',
            'Disaster recovery',
            'DevSecOps practices'
        ],
        'expert': [
            'Platform engineering',
            'Cloud-native architecture',
            'Enterprise DevOps strategy',
            'FinOps and cost optimization',
            'Team leadership',
            'Chaos engineering',
            'DevOps culture transformation'
        ]
    },
    'UX Designer': {
        'foundation': [
            'Design principles and theory',
            'User research basics',
            'Wireframing and prototyping',
            'UI fundamentals',
            'Design tools (Figma, Sketch)',
            'Information architecture',
            'Basic HTML/CSS'
        ],
        'intermediate': [
            'User research methods',
            'Interaction design patterns',
            'Usability testing',
            'Design systems',
            'Responsive design',
            'Accessibility standards',
            'User journey mapping'
        ],
        'advanced': [
            'Advanced prototyping',
            'Design strategy',
            'Service design',
            'Design leadership',
            'A/B testing',
            'Design ops',
            'Cross-platform design'
        ],
        'expert': [
            'UX strategy and vision',
            'Design innovation',
            'Team leadership',
            'Enterprise design systems',
            'Design thinking workshops',
            'Industry speaking',
            'Mentorship programs'
        ]
    },
    'Cybersecurity Analyst': {
        'foundation': [
            'Networking fundamentals',
            'Operating systems (Linux/Windows)',
            'Security concepts and principles',
            'Basic scripting (Python, Bash)',
            'Cryptography basics',
            'Security tools and utilities',
            'Risk management fundamentals'
        ],
        'intermediate': [
            'Network security',
            'Security operations',
            'Incident response',
            'Vulnerability assessment',
            'Security frameworks',
            'Threat intelligence',
            'Digital forensics basics'
        ],
        'advanced': [
            'Penetration testing',
            'Advanced threat hunting',
            'Security architecture',
            'Cloud security',
            'Malware analysis',
            'Security automation',
            'Compliance and regulations'
        ],
        'expert': [
            'Security strategy',
            'Advanced forensics',
            'Zero trust architecture',
            'Security program management',
            'Team leadership',
            'Security consulting',
            'Industry research'
        ]
    },
    'Cloud Architect': {
        'foundation': [
            'Cloud fundamentals (AWS/Azure/GCP)',
            'Networking basics',
            'Linux administration',
            'Virtualization concepts',
            'Basic scripting',
            'Security fundamentals',
            'Cost management basics'
        ],
        'intermediate': [
            'Cloud services and solutions',
            'Infrastructure as Code',
            'Container orchestration',
            'Cloud security patterns',
            'Microservices architecture',
            'Disaster recovery',
            'Performance optimization'
        ],
        'advanced': [
            'Multi-cloud strategy',
            'Cloud-native architecture',
            'Enterprise cloud migration',
            'Advanced security',
            'FinOps practices',
            'Service mesh',
            'Serverless architecture'
        ],
        'expert': [
            'Enterprise architecture',
            'Cloud strategy leadership',
            'Innovation and R&D',
            'Team management',
            'Vendor management',
            'Industry consulting',
            'Cloud governance'
        ]
    },
    'AI/ML Engineer': {
        'foundation': [
            'Python programming',
            'Mathematics and statistics',
            'Machine learning basics',
            'Data structures',
            'SQL and databases',
            'Version control',
            'Development tools'
        ],
        'intermediate': [
            'Deep learning frameworks',
            'Neural network architectures',
            'Model training and validation',
            'Feature engineering',
            'MLOps basics',
            'Data pipeline development',
            'Model optimization'
        ],
        'advanced': [
            'Advanced deep learning',
            'Reinforcement learning',
            'Computer vision',
            'NLP and transformers',
            'Distributed training',
            'Model deployment at scale',
            'Research methods'
        ],
        'expert': [
            'AI research leadership',
            'Novel architecture design',
            'AI ethics and governance',
            'Team management',
            'Industry consulting',
            'Publication and patents',
            'AI strategy development'
        ]
    },
    'Product Manager': {
        'foundation': [
            'Product development lifecycle',
            'Agile methodologies',
            'User story writing',
            'Basic analytics',
            'Stakeholder management',
            'Market research basics',
            'Technical fundamentals'
        ],
        'intermediate': [
            'Product strategy',
            'Data-driven decision making',
            'User research methods',
            'Competitive analysis',
            'Product metrics',
            'Roadmap planning',
            'A/B testing'
        ],
        'advanced': [
            'Product vision and strategy',
            'Growth strategies',
            'Advanced analytics',
            'Team leadership',
            'Product operations',
            'Stakeholder management',
            'Go-to-market strategy'
        ],
        'expert': [
            'Portfolio management',
            'Enterprise product strategy',
            'Innovation leadership',
            'Executive communication',
            'Organizational leadership',
            'Industry thought leadership',
            'Product organization design'
        ]
    }
} 