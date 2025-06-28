document.addEventListener('DOMContentLoaded', function() {
    const reSlider = document.getElementById('re-slider');
    const reValue = document.getElementById('re-value');
    const airfoilSelect = document.getElementById('airfoil-select');
    const viewButtons = document.querySelectorAll('.view-btn');
    const flowVisualization = document.getElementById('flow-visualization');
    const loadingSpinner = document.getElementById('loading-spinner');
    const maxVelocity = document.getElementById('max-velocity');
    const minPressure = document.getElementById('min-pressure');
    const maxPressure = document.getElementById('max-pressure');
    const demoModeWarning = document.getElementById('demo-mode-warning');
    
    let currentRe = Math.pow(10, parseFloat(reSlider.value));
    let currentAirfoil = airfoilSelect.value;
    let currentView = 'velocity';
    let predictionInProgress = false;
    let debounceTimer;
    
    updateReValue();
    fetchPrediction();
    
    fetchAirfoilOptions();
    
    reSlider.addEventListener('input', function() {
        updateReValue();
    });
    
    reSlider.addEventListener('change', function() {
        fetchPrediction();
    });
    
    reSlider.addEventListener('input', debouncedFetchPrediction);
    
    airfoilSelect.addEventListener('change', function() {
        currentAirfoil = airfoilSelect.value;
        fetchPrediction();
    });
    
    viewButtons.forEach(button => {
        button.addEventListener('click', function() {
            viewButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            currentView = this.dataset.view;
            fetchPrediction();
        });
    });
    
    function updateReValue() {
        const sliderValue = parseFloat(reSlider.value);
        currentRe = Math.pow(10, sliderValue);
        reValue.textContent = formatNumber(currentRe);
    }
    
    function debouncedFetchPrediction() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(fetchPrediction, 300);
    }
    
    function fetchPrediction() {
        if (predictionInProgress) return;
        
        predictionInProgress = true;
        loadingSpinner.style.display = 'block';
        flowVisualization.style.opacity = '0.5';
        
        const url = `/api/predict?re=${currentRe}&airfoil=${currentAirfoil}&output_type=${currentView}`;
        
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                flowVisualization.src = `data:image/png;base64,${data.visualization}`;
                
                maxVelocity.textContent = data.max_velocity.toFixed(3);
                minPressure.textContent = data.min_pressure.toFixed(3);
                maxPressure.textContent = data.max_pressure.toFixed(3);
                
                if (data.demo_mode) {
                    demoModeWarning.style.display = 'block';
                } else {
                    demoModeWarning.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error fetching prediction:', error);
                alert('Error fetching prediction. Please try again.');
            })
            .finally(() => {
                predictionInProgress = false;
                loadingSpinner.style.display = 'none';
                flowVisualization.style.opacity = '1';
            });
    }
    
    function fetchAirfoilOptions() {
        fetch('/api/airfoils')
            .then(response => response.json())
            .then(data => {
                airfoilSelect.innerHTML = '';
                data.airfoils.forEach(airfoil => {
                    const option = document.createElement('option');
                    option.value = airfoil.id;
                    option.textContent = `${airfoil.name} - ${airfoil.description}`;
                    airfoilSelect.appendChild(option);
                });
                
                airfoilSelect.value = currentAirfoil;
            })
            .catch(error => {
                console.error('Error fetching airfoil options:', error);
            });
    }
    
    function formatNumber(number) {
        return number.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
});
