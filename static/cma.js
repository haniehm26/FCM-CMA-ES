function showTable(data) {
    // Show the table container
    document.getElementById("table-container").style.display = "block";
    var n_generation = document.getElementById("n-generation")
    var n_offspring = document.getElementById("n-offspring")
    var n_parent = document.getElementById("n-parent")
    var X = document.getElementById("X")
    var y = document.getElementById("y")
    var execTime = document.getElementById("exec-time")
    n_generation.textContent = data.n_offspring;
    n_offspring.textContent = data.n_generation;
    n_parent.textContent = data.n_parent;
    X.textContent = data.X;
    y.textContent = data.y;
    execTime.textContent = data.exec_time;
}

function sendDataToServer(data) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/cma", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            // Hide loading indicator
            var loadingIndicator = document.getElementById("loading-indicator");
            loadingIndicator.style.display = "none";
            if (xhr.status === 200) {
                var result = JSON.parse(xhr.responseText);
                showTable(result.table_data);
                if (data.dimension == 2) {
                    showChart(result.chart_data);
                }
            } else {
                // Request encountered an error, handle the error case
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
    document.getElementById("table-container").style.display = "none";
    const parentElement = document.getElementById("img-container");
    // Remove all children
    while (parentElement.firstChild) {
        parentElement.firstChild.remove();
    }
    parentElement.style.display = "none";
    // Retrieve form data
    var dimension = document.getElementById("dimension").value;
    var benchmarkFunction = document.getElementById("benchmark-function").value;
    var data = {
        dimension: dimension,
        benchmark_function: benchmarkFunction
    };
    // Send the data to the server
    sendDataToServer(data);
});