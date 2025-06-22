document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('careerForm');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const recommendationsContainer = document.getElementById('recommendations');

    // Update range input values
    document.querySelectorAll('input[type="range"]').forEach(input => {
        const valueDisplay = input.nextElementSibling;
        input.addEventListener('input', () => {
            valueDisplay.textContent = input.value + (input.id.includes('percentage') ? '%' : '');
        });
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading spinner
        loading.style.display = 'block';
        results.style.display = 'none';
        
        // Collect form data
        const formData = new FormData(form);
        const data = {
            skills: {
                programming_concept_percentage: parseInt(formData.get('programming_concept_percentage')),
                communication_skills_percentage: parseInt(formData.get('communication_skills_percentage')),
                coding_skills_rating: parseInt(formData.get('coding_skills_rating')),
                public_speaking_points: parseInt(formData.get('public_speaking_points')),
                self_learning_capability: parseInt(formData.get('self_learning_capability'))
            },
            experience: {
                hours_working_per_day: parseInt(formData.get('hours_working_per_day')),
                hackathons: parseInt(formData.get('hackathons')),
                certifications: parseInt(formData.get('certifications')),
                workshops: parseInt(formData.get('workshops'))
            },
            interests: {
                interested_subjects: Array.from(formData.getAll('interested_subjects')),
                interested_career_area: Array.from(formData.getAll('interested_career_area')),
                company_type: formData.get('company_type')
            }
        };

        try {
            console.log('Sending data:', data);
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const responseData = await response.json();
            console.log('Received response:', responseData);

            if (!response.ok) {
                throw new Error(responseData.details || 'Failed to get recommendations');
            }

            if (responseData.error) {
                throw new Error(responseData.error);
            }

            // Store recommendations in sessionStorage
            sessionStorage.setItem('careerRecommendations', JSON.stringify(responseData));
            
            // Redirect to results page
            window.location.href = '/results';

        } catch (error) {
            console.error('Error:', error);
            displayError(error.message || 'Failed to get career recommendations. Please try again.');
        } finally {
            loading.style.display = 'none';
        }
    });

    function displayError(message) {
        results.style.display = 'block';
        if (recommendationsContainer) {
            recommendationsContainer.innerHTML = `
                <div class="error-message">
                    <h3>Error</h3>
                    <p>${message}</p>
                </div>
            `;
        }
    }

    // Mobile menu toggle
    const menuToggle = document.querySelector('.menu-toggle');
    const nav = document.querySelector('nav ul');
    
    if (menuToggle && nav) {
        menuToggle.addEventListener('click', () => {
            nav.classList.toggle('show');
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}); 