// static/script.js
document.addEventListener("DOMContentLoaded", function() {
    const predictBtn = document.getElementById("predict-btn");
    const predictionOutput = document.getElementById("prediction-output");
    const pastPredictions = document.getElementById("past-predictions");
  
    predictBtn.addEventListener("click", function() {
      fetch("/api/prediction")
        .then(response => response.json())
        .then(data => {
          predictionOutput.textContent = data.prediction;
          loadPastPredictions();
        })
        .catch(err => {
          predictionOutput.textContent = "Error: " + err;
        });
    });
  
    function loadPastPredictions() {
      fetch("/api/predictions")
        .then(response => response.json())
        .then(data => {
          pastPredictions.innerHTML = "";
          data.forEach(pred => {
            const div = document.createElement("div");
            div.textContent = pred.prediction;
            pastPredictions.appendChild(div);
          });
        })
        .catch(err => {
          pastPredictions.textContent = "Error: " + err;
        });
    }
  
    // Load past predictions on page load
    loadPastPredictions();
  });