function initializeMovableButton(lineContainerId, movableButtonId, currentValueId, startToggleId, endToggleId, minValue, maxValue, stepSize) {
    var movableButton = document.getElementById(movableButtonId);
    var lineContainer = document.getElementById(lineContainerId);
    var valueDisplay = document.getElementById(currentValueId);
    var startToggle = document.getElementById(startToggleId);
    var endToggle = document.getElementById(endToggleId);
    var lineContainerRect = lineContainer.getBoundingClientRect();
    var isDragging = false;

    movableButton.addEventListener("mousedown", startDragging);
    document.addEventListener("mousemove", moveButton);
    document.addEventListener("mouseup", stopDragging);

    startToggle.addEventListener("click", moveOneUnitLeft);
    endToggle.addEventListener("click", moveOneUnitRight);

    function startDragging(event) {
        isDragging = true;
    }

    function moveButton(event) {
        if (!isDragging) return;

        var mouseX = event.clientX - lineContainerRect.left;
        var maxX = lineContainerRect.width;

        var positionRatio = mouseX / maxX; // Calculate the ratio of mouse position to the maximum position

        var value = positionRatio * (maxValue - minValue) + minValue; // Calculate the corresponding value

        var roundedValue;

        roundedValue = Math.round(value / stepSize) * stepSize; // Round the value to the nearest step

        // Check if the rounded value exceeds the maxValue and adjust if necessary
        if (roundedValue > maxValue) {
            roundedValue = maxValue;
        }

        if (roundedValue < minValue) {
            roundedValue = minValue;
        }

        var newPosition = (roundedValue - minValue) / (maxValue - minValue) * maxX; // Calculate the new position based on the rounded value

        movableButton.style.left = newPosition + "px";
        valueDisplay.textContent = roundedValue;
    }

    function stopDragging() {
        isDragging = false;
    }

    function moveOneUnitLeft() {
        var currentPosition = parseFloat(movableButton.style.left) || 0;

        var step = (lineContainerRect.width / (maxValue - minValue)) * stepSize;

        var newPosition = Math.max(0, currentPosition - step);
        movableButton.style.left = newPosition + "px";

        var value = Math.round((newPosition / lineContainerRect.width) * (maxValue - minValue) + minValue);
        valueDisplay.textContent = value;
    }

    function moveOneUnitRight() {
        var currentPosition = parseFloat(movableButton.style.left) || 0;

        var step = (lineContainerRect.width / (maxValue - minValue)) * stepSize;

        var newPosition = Math.min(lineContainerRect.width, currentPosition + step);
        movableButton.style.left = newPosition + "px";

        var value = Math.round((newPosition / lineContainerRect.width) * (maxValue - minValue) + minValue);
        valueDisplay.textContent = value;
    }
}

function showChart(data) {
    displayImage(data.cma_contour);
    displayImage(data.sigma_chart);
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

function sendDataToServer(data, dimension) {
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
                if (dimension == 2) {
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
initializeMovableButton("dimension-line-container", "dimension-movable-button", "dimension-current-value", "dimension-start-toggle", "dimension-end-toggle", 2, 32, 1);

document.getElementById("form").addEventListener("submit", function (event) {
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
    var dimension = document.getElementById("dimension-current-value").textContent;
    var benchmarkFunction = document.getElementById("benchmark-function").value;

    var data = {
        dimension: dimension,
        benchmark_function: benchmarkFunction
    };

    // Send the data to the server
    sendDataToServer(data, dimension);
});