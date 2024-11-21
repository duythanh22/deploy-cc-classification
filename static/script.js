document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('image-upload');
    const toggleButton = document.getElementById('toggleProbabilities');
    const probabilitiesContent = document.getElementById('probabilitiesContent');
    const loadingIndicator = document.getElementById('loading');
    const resultsSection = document.getElementById('results');
    const errorSection = document.getElementById('error');
    const maxFileSizeMB = 10; // Set the file size limit to 10MB
    let chartInstance = null;  // Store chart instance to reset it later

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const file = fileInput.files[0];

        // File size validation
        if (file && file.size > maxFileSizeMB * 1024 * 1024) {
            errorSection.classList.remove('hidden');
            errorSection.querySelector('p').textContent = `File size exceeds the 10MB limit. Please choose a smaller file.`;
            return; // Stop the form submission
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
        .then(response => response.json())
        .then(data => {
            loadingIndicator.classList.add('hidden');
            if (data.error) {
                errorSection.classList.remove('hidden');
                errorSection.querySelector('p').textContent = data.error;
            } else {
                // Extract keys and values from the class_probabilities object
                const classLabels = Object.keys(data.class_probabilities);
                const probabilities = Object.values(data.class_probabilities);

                const confidenceScore = data.confidence_score;  // Độ tự tin từ API
                const confidenceThreshold = 0.6; // 60% confidence threshold

                const predictionElement = document.getElementById('prediction');
                const confidenceElement = document.getElementById('confidence');

                predictionElement.textContent = data.predicted_class;
                confidenceElement.textContent = `${(confidenceScore * 100).toFixed(2)}%`;  // Sử dụng confidence_score từ API

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

                // Reset and draw the chart with actual data
                drawProbabilityChart(classLabels, probabilities);
            }
        })

        .catch(error => {
            loadingIndicator.classList.add('hidden');
            errorSection.classList.remove('hidden');
            errorSection.querySelector('p').textContent = 'An error occurred while processing your request.';
            console.error('Error:', error);
        });
    });

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
