function showChart(data) {
    displayImage(data.es_total);
    displayImage(data.es_best);
    displayImage(data.es_total_contour);
    displayImage(data.es_best_contour);
}

function displayImage(base64Data) {
    const imageContainer = document.getElementById('img-container');
    const imageElement = document.createElement('img');
    imageElement.src = 'data:image/png;base64,' + base64Data;
    imageContainer.appendChild(imageElement);
    imageContainer.style.display = "block"
}

function showTable(data) {
    // Show the table container
    document.getElementById("table-container").style.display = "block";
    var X = document.getElementById("X")
    var y = document.getElementById("y")
    var execTime = document.getElementById("exec-time")
    X.textContent = data.X;
    y.textContent = data.y;
    execTime.textContent = data.exec_time;
}

function sendDataToServer(data) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/es", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            // Hide loading indicator
            var loadingIndicator = document.getElementById("loading-indicator");
            loadingIndicator.style.display = "none";
            if (xhr.status === 200) {
                var errorIndicator = document.getElementById("error-indicator");
                errorIndicator.style.display = "none";
                var result = JSON.parse(xhr.responseText);
                showTable(result.table_data);
                if (data.dimension == 2) {
                    showChart(result.chart_data);
                }
            } else {
                // Request encountered an error, handle the error case
                var errorIndicator = document.getElementById("error-indicator");
                errorIndicator.style.display = "block";
                console.error("Error:", xhr.status);
            }
        }
    };
    xhr.send(JSON.stringify(data));
}

// Call the functions
document.getElementById("submit").addEventListener("click", function (event) {
    event.preventDefault(); // Prevent form submission
    // Display loading indicator
    document.getElementById("loading-indicator").style.display = "block";
    document.getElementById("error-indicator").style.display = "none";
    document.getElementById("table-container").style.display = "none";
    const parentElement = document.getElementById("img-container");
    // Remove all children
    while (parentElement.firstChild) {
        parentElement.firstChild.remove();
    }
    parentElement.style.display = "none";
    // Retrieve form data
    var dimension = document.getElementById("dimension").value;
    var generationSize = document.getElementById("generation-size").value;
    var offspringSize = document.getElementById("offspring-size").value;
    var parentSize = document.getElementById("parent-size").value;
    var stepSize = document.getElementById("step-size").value;
    var survivorSelection = document.getElementById("survivor-selection").value;
    var benchmarkFunction = document.getElementById("benchmark-function").value;
    var data = {
        dimension: dimension,
        generation_size: generationSize,
        offspring_size: offspringSize,
        parent_size: parentSize,
        step_size: stepSize,
        survivor_selection: survivorSelection,
        benchmark_function: benchmarkFunction
    };
    // Send the data to the server
    sendDataToServer(data);
});