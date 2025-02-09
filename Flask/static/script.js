document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const imageContainer = document.getElementById('image-container');
    const resultDiv = document.getElementById('prediction-result');
    const navLinks = document.querySelectorAll('.nav-link');
  
    navLinks.forEach(link => {
      link.addEventListener('click', function() {
        const target = this.getAttribute('href').substring(1);
        if (target === 'test') {
          form.style.display = 'block';
          imageContainer.style.display = 'block';
          resultDiv.style.display = 'block';
        } else {
          form.style.display = 'none';
          imageContainer.style.display = 'none';
          resultDiv.style.display = 'none';
        }
      });
    });
  
    form.addEventListener('submit', function(event) {
      event.preventDefault();
      const imageInput = document.getElementById('image-input');
      const file = imageInput.files[0];
      if (!file) {
        alert('Please select an image file');
        return;
      }
  
      const formData = new FormData();
      formData.append('file', file);
      fetch('/predict', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        const imageUrl = URL.createObjectURL(file);
        const imageElement = document.createElement('img');
        imageElement.src = imageUrl;
        imageContainer.innerHTML = '';
        imageContainer.appendChild(imageElement);
        resultDiv.innerHTML = '<h2>Predictions:</h2>';
        const predictions = data.prediction;
        const confidences = data.confidences;
        predictions.forEach(prediction => {
          const confidence = confidences[prediction];
          const p = document.createElement('p');
          p.innerHTML = `Class: <strong>${prediction}</strong>, Confidence: <strong>${(confidence * 100).toFixed(2)}%</strong>`;
          resultDiv.appendChild(p);
        });
      })
      .catch(error => {
        console.error('Error:', error);
        resultDiv.textContent = 'Error occurred during prediction';
      });
    });
  });