function showChart(data) {
    displayImage(data.dataset_png);
    displayImage(data.algorithm_png);
    displayImage(data.cost_function);
    displayImage(data.l_param);
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
    var rand_index = document.getElementById("rand-index")
    var fc = document.getElementById("fc")
    var hc = document.getElementById("hc")
    var cost_value = document.getElementById("cost-value")
    var exec_time = document.getElementById("exec-time")
    rand_index.textContent = data.rand_index;
    fc.textContent = data.fc;
    hc.textContent = data.hc;
    cost_value.textContent = data.cost_value;
    exec_time.textContent = data.exec_time;
}

function sendDataToServer(data) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/fcm-cma", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            // Hide loading indicator
            var loadingIndicator = document.getElementById("loading-indicator");
            loadingIndicator.style.display = "none";
            if (xhr.status === 200) {
                var result = JSON.parse(xhr.responseText);
                showTable(result.table_data);
                showChart(result.chart_data);
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
    var m = document.getElementById("m-param").value;
    var l = document.getElementById("l-param").value;
    var dataset = document.getElementById("dataset").value;
    var data = {
        m: m,
        l: l,
        dataset_name: dataset
    };
    // Send the data to the server
    sendDataToServer(data);
});