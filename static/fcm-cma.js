function initializeMovableButton(lineContainerId, movableButtonId, currentValueId, startToggleId, endToggleId, minValue, maxValue, isFloat, stepSize) {
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

        if (isFloat) {
            roundedValue = (Math.round(value / stepSize) * stepSize).toFixed(2); // Round the value to 2 decimal places
        } else {
            roundedValue = Math.round(value / stepSize) * stepSize; // Round the value to the nearest step
        }

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

        if (isFloat) {
            var step = ((lineContainerRect.width) / (maxValue - minValue)) * stepSize;
        }
        else {
            var step = (lineContainerRect.width / (maxValue - minValue)) * stepSize;
        }

        var newPosition = Math.max(0, currentPosition - step);
        movableButton.style.left = newPosition + "px";

        if (isFloat) {
            var value = (newPosition / lineContainerRect.width) * (maxValue - minValue) + minValue;
            value = value.toFixed(2);
        }
        else {
            var value = Math.round((newPosition / lineContainerRect.width) * (maxValue - minValue) + minValue);
        }
        valueDisplay.textContent = value;
    }

    function moveOneUnitRight() {
        var currentPosition = parseFloat(movableButton.style.left) || 0;

        if (isFloat) {
            var step = ((lineContainerRect.width) / (maxValue - minValue)) * stepSize;
        }
        else {
            var step = (lineContainerRect.width / (maxValue - minValue)) * stepSize;
        }

        var newPosition = Math.min(lineContainerRect.width, currentPosition + step);
        movableButton.style.left = newPosition + "px";

        if (isFloat) {
            var value = (newPosition / lineContainerRect.width) * (maxValue - minValue) + minValue;
            value = value.toFixed(2);
        } else {
            var value = Math.round((newPosition / lineContainerRect.width) * (maxValue - minValue) + minValue);
        }
        valueDisplay.textContent = value;
    }
}

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
initializeMovableButton("n-iter-line-container", "n-iter-movable-button", "n-iter-current-value", "n-iter-start-toggle", "n-iter-end-toggle", 100, 10000, false, 50);
initializeMovableButton("m-line-container", "m-movable-button", "m-current-value", "m-start-toggle", "m-end-toggle", 2, 101, true, 0.5);

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
    var nIter = document.getElementById("n-iter-current-value").textContent;
    var m = document.getElementById("m-current-value").textContent;
    var l = document.getElementById("l-param").value;
    var dataset = document.getElementById("dataset").value;

    var data = {
        n_iter: nIter,
        m: m,
        l: l,
        dataset_name: dataset
    };

    // Send the data to the server
    sendDataToServer(data);
});