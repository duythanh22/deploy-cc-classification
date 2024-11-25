document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('image-upload');
    const toggleButton = document.getElementById('toggleProbabilities');
    const probabilitiesContent = document.getElementById('probabilitiesContent');
    const loadingIndicator = document.getElementById('loading');
    const resultsSection = document.getElementById('results');
    const errorSection = document.getElementById('error');
    const maxFileSizeMB = 10;
    let chartInstance = null;

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const file = fileInput.files[0];

        // File size validation
        if (file && file.size > maxFileSizeMB * 1024 * 1024) {
            showError(`File size exceeds the 10MB limit. Please choose a smaller file.`);
            return;
        }

        // File type validation
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (file && !allowedTypes.includes(file.type)) {
            showError(`Unsupported file type. Please upload a JPEG or PNG image.`);
            return;
        }

        const formData = new FormData(form);

        loadingIndicator.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        probabilitiesContent.classList.add('hidden');
        toggleButton.textContent = 'Show Probabilities';

        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const uploadedImage = document.getElementById('uploadedImage');
                const imagePreview = document.getElementById('imagePreview');
                uploadedImage.src = e.target.result;
                imagePreview.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {

                if (response.status === 413) {
                    throw new Error('Payload too large');
                }
                return response.json().then(err => {
                    throw new Error(err.detail || 'Server error occurred');
                });
            }
            return response.json();
        })
        .then(data => {
            loadingIndicator.classList.add('hidden');

            if (!data || data.length === 0) {
                throw new Error('No data received from server');
            }

            if (data.detail) {
                throw new Error(data.detail);
            }

            const result = Array.isArray(data) ? data[0] : data;

            if (result.error) {
                showError(result.error);
                return;
            }

            if (!result.predicted_class || !result.class_probabilities || !result.confidence_score) {
                throw new Error('Invalid response format from server');
            }

            const classLabels = Object.keys(result.class_probabilities);
            const probabilities = Object.values(result.class_probabilities);
            const confidenceScore = result.confidence_score;
            const confidenceThreshold = 0.6;

            const predictionElement = document.getElementById('prediction');
            const confidenceElement = document.getElementById('confidence');

            predictionElement.textContent = result.predicted_class;
            confidenceElement.textContent = `${(confidenceScore * 100).toFixed(2)}%`;

            while (predictionElement.childNodes.length > 1) {
                predictionElement.removeChild(predictionElement.lastChild);
            }

            if (confidenceScore < confidenceThreshold) {
                predictionElement.style.color = 'red';
                const annotation = document.createElement('p');
                annotation.classList.add('text-sm', 'text-red-500', 'mt-2');
                annotation.textContent = 'Warning: The confidence in this prediction is low (less than 60%). Please check the probabilities of all labels below.';
                predictionElement.appendChild(annotation);
            } else {
                predictionElement.style.color = 'green';
            }

            resultsSection.classList.remove('hidden');

            if (typeof drawProbabilityChart === 'function') {
                drawProbabilityChart(classLabels, probabilities);
            }
        })
        .catch(error => {
            loadingIndicator.classList.add('hidden');
            let errorMessage = 'An error occurred while processing your request.';

            if (error.message.includes('Unsupported image format')) {
                errorMessage = 'Please upload only JPEG or PNG images.';
            } else if (error.message.includes('No data received')) {
                errorMessage = 'No response received from the server. Please try again.';
            } else if (error.message.includes('Invalid response format')) {
                errorMessage = 'Server returned an invalid response format. Please try again.';
            } else if (error.message.includes('Server error')) {
                errorMessage = 'A server error occurred. Please try again later.';
            } else if (error.message.includes('Payload too large')) {
                errorMessage = 'The uploaded file is too large. Please choose a smaller file.';
            }
            showError(errorMessage);
            console.error('Error:', error);
        });
    });

    // Helper function to show error messages
    function showError(message) {
        errorSection.classList.remove('hidden');
        errorSection.querySelector('p').textContent = message;
        loadingIndicator.classList.add('hidden');
        resultsSection.classList.add('hidden');
    }

    if (toggleButton && probabilitiesContent) {
        toggleButton.addEventListener('click', function() {
            probabilitiesContent.classList.toggle('hidden');
            toggleButton.textContent = probabilitiesContent.classList.contains('hidden') ?
                'Show Probabilities' : 'Hide Probabilities';
        });
    }

    const dropZone = document.querySelector('.drop-zone');

    if (dropZone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            dropZone.classList.add('border-blue-500');
        }

        function unhighlight(e) {
            dropZone.classList.remove('border-blue-500');
        }

        dropZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            updateFileName(files[0].name);

            // Check file size on drop
            if (files[0].size > maxFileSizeMB * 1024 * 1024) {
                errorSection.classList.remove('hidden');
                errorSection.querySelector('p').textContent = `File size exceeds the 10MB limit. Please choose a smaller file.`;
            } else {
                errorSection.classList.add('hidden');
            }
        }

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                updateFileName(e.target.files[0].name);

                // Check file size on file input change
                if (e.target.files[0].size > maxFileSizeMB * 1024 * 1024) {
                    errorSection.classList.remove('hidden');
                    errorSection.querySelector('p').textContent = `File size exceeds the 10MB limit. Please choose a smaller file.`;
                } else {
                    errorSection.classList.add('hidden');
                }
            }
        });

        function updateFileName(fileName) {
            dropZone.querySelector('p').textContent = `Selected file: ${fileName}`;
        }
    }

    function drawProbabilityChart(labels, data) {
        const ctx = document.getElementById('probabilityChart').getContext('2d');

        // Destroy the old chart instance if it exists
        if (chartInstance) {
            chartInstance.destroy();
        }

        // Define different colors for each label
        const backgroundColors = [
            '#4CAF50', '#FF5722', '#FFC107', '#03A9F4', '#9C27B0',
            '#E91E63', '#FF9800', '#009688', '#8BC34A', '#CDDC39'
        ];

        // Select colors for bars dynamically based on the number of labels
        const barColors = labels.map((_, index) => backgroundColors[index % backgroundColors.length]);

        // Generate a new chart with customized colors for each bar
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: data,
                    backgroundColor: barColors, // Apply different colors
                    borderColor: '#ffffff',
                    borderWidth: 1,
                    hoverBackgroundColor: '#fe7903',
                    hoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        // text: 'Biểu đồ Xác Suất Dự Đoán',
                        font: { size: 18, weight: 'bold' },
                        padding: { top: -30, bottom: 20 }
                    },
                    tooltip: {
                        enabled: true,
                        callbacks: {
                            label: function(tooltipItem) {
                                return ` Xác suất: ${tooltipItem.raw}%`;
                            }
                        }
                    },
                    legend: {
                        display: true,
                        labels: {
                            font: { size: 14 },
                            color: '#000'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#000',
                            font: { size: 12 }
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 10,
                            color: '#000',
                            font: { size: 12 }
                        },
                        grid: {
                            borderDash: [5, 5],
                            color: '#e0e0e0'
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutBounce'
                }
            }
        });
    }
});
