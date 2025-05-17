document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Predicting...';
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.predictions) {
            // Show prediction in bold, uppercase, and with a custom color
            const pred = data.predictions[0].toUpperCase();
            resultDiv.innerHTML = `<b style="color:#ffb300; font-size:1.4em;">${pred}</b>`;
        } else if (data.error) {
            resultDiv.textContent = 'Error: ' + data.error;
        } else {
            resultDiv.textContent = 'Unknown error.';
        }
    } catch (err) {
        resultDiv.textContent = 'Error: ' + err.message;
    }
});
