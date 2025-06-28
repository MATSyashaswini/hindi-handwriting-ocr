document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM Loaded. Initializing OCR script...");

    // --- CONFIGURATION ---
    const API_PREDICT_ENDPOINT = '/predict';
    const API_EXAMPLES_ENDPOINT = '/api/examples';
    const ALLOWED_MIMETYPES = ['image/png', 'image/jpeg', 'image/jpg'];
    // ---------------------

    // --- DOM Elements (Ensure these IDs exist in recognizer.html) ---
    const imageUploadInput = document.getElementById('imageUpload');
    const fileNameSpan = document.getElementById('fileName');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const predictButton = document.getElementById('predictButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultDisplay = document.getElementById('resultDisplay');
    const predictionText = document.getElementById('predictionText');
    const probabilityText = document.getElementById('probabilityText'); // Element for probability
    const errorDisplay = document.getElementById('errorDisplay');
    const errorText = document.getElementById('errorText');
    const examplesContainer = document.getElementById('examplesContainer');

    const isRecognizerPage = !!imageUploadInput && !!predictButton;
    console.log("Is Recognizer Page:", isRecognizerPage);

    let currentFile = null;
    let currentObjectURL = null;

    // --- Initial Setup ---
    if (isRecognizerPage) {
        console.log("On Recognizer Page: Loading examples and setting up listeners.");
        loadExampleImages();
        setupEventListeners();
    } else {
        console.log("Not on recognizer page, skipping OCR-specific setup.");
    }

    // --- Setup Event Listeners Function ---
    function setupEventListeners() {
        if (imageUploadInput) {
            imageUploadInput.addEventListener('change', handleFileSelect);
            console.log("Added listener for file input change.");
        }
        if (predictButton) {
            predictButton.addEventListener('click', handlePrediction);
            console.log("Added listener for predict button click.");
        }
        document.querySelectorAll('.custom-switch-input').forEach(switchInput => {
             switchInput.addEventListener('change', (event) => {
                 console.log('Switch changed state:', event.target.checked);
             });
        });
    }

    // --- Core Functions ---

    async function loadExampleImages() {
        if (!examplesContainer) {
            console.warn("Examples container not found, cannot load examples.");
            return;
        }
        console.log("Fetching example images from:", API_EXAMPLES_ENDPOINT);
        examplesContainer.innerHTML = `
            <div class="col-span-full text-center p-4">
                <div class="inline-flex items-center text-gray-500">
                    <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    Loading examples...
                </div>
            </div>`;

        try {
            const response = await fetch(API_EXAMPLES_ENDPOINT);
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
            }
            const examples = await response.json();

            examplesContainer.innerHTML = '';

            if (Array.isArray(examples) && examples.length > 0) {
                console.log(`Found ${examples.length} examples.`);
                examples.forEach(example => {
                    if (!example.url || !example.filename) {
                        console.warn("Skipping invalid example item:", example);
                        return;
                    }
                    const container = document.createElement('div');
                    container.className = 'aspect-w-3 aspect-h-2 overflow-hidden rounded-md border border-gray-200 hover:border-primary transition duration-150 ease-in-out shadow-sm bg-gray-50';
                    const img = document.createElement('img');
                    img.src = example.url;
                    img.alt = `Example: ${example.filename}`;
                    img.title = `Click to use ${example.filename}`;
                    img.className = 'w-full h-full object-contain cursor-pointer p-1';
                    img.loading = 'lazy';
                    img.addEventListener('click', () => handleExampleSelect(example));
                    img.onerror = () => {
                        console.error(`Failed to load example image: ${example.url}`);
                        img.alt = `Error loading ${example.filename}`;
                        container.innerHTML = `<div class="w-full h-full flex items-center justify-center text-red-500 text-xs p-1">${img.alt}</div>`;
                    }
                    container.appendChild(img);
                    examplesContainer.appendChild(container);
                });
            } else {
                console.log("No examples found or received empty/invalid data.");
                examplesContainer.innerHTML = '<div class="col-span-full text-gray-500 text-sm p-4 text-center">No example images available.</div>';
            }
        } catch (error) {
            console.error("Failed to load example images:", error);
            examplesContainer.innerHTML = `<div class="col-span-full text-red-600 text-sm p-4 text-center">Could not load examples: ${error.message}</div>`;
        }
    }

    function handleFileSelect(event) {
        const file = event.target?.files?.[0];
        if (file) {
            console.log(`File selected via input: ${file.name}`);
            resetUI(false);
            processSelectedFile(file, 'upload');
        } else {
            console.log("File selection event triggered, but no file selected or cancelled.");
            if (!currentFile) {
                resetUI(true);
            }
        }
    }

    async function handleExampleSelect(example) {
        if (!example?.url || !example?.filename) {
            console.error("Invalid example data provided:", example);
            showError("Invalid example selected.");
            return;
        }
        console.log(`Example selected: ${example.filename}`);
        resetUI(false);
        displayFileName(example.filename);
        showImagePreview(example.url);

        try {
            showTemporaryLoading("Preparing example image...");
            if (predictButton) predictButton.disabled = true;
            const response = await fetch(example.url);
            if (!response.ok) {
                throw new Error(`Failed to fetch image (${response.status})`);
            }
            const blob = await response.blob();
            const fetchedFile = new File([blob], example.filename, { type: blob.type });
            fetchedFile.previewURL = example.url;
            hideTemporaryLoading();
            processSelectedFile(fetchedFile, 'example');
        } catch (error) {
            console.error("Error fetching example image blob:", error);
            showError(`Could not load example image: ${error.message}`);
            resetUI(true);
        }
    }

    function processSelectedFile(file, sourceType) {
        if (!file) {
             console.warn("processSelectedFile called without a valid file object.");
             return;
        }
        if (!ALLOWED_MIMETYPES.includes(file.type)) {
            showError(`Invalid file type: '${file.type}'. Please use PNG, JPG, or JPEG.`);
            resetUI(true);
            return;
        }
        console.log(`Processing file: ${file.name}, Type: ${file.type}, Size: ${file.size} bytes, Source: ${sourceType}`);
        currentFile = file;
        if (sourceType === 'upload') {
             displayFileName(file.name);
        }
        if (sourceType === 'upload') {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (e.target?.result) {
                    showImagePreview(e.target.result);
                } else {
                     showError("Could not read file for preview.");
                     resetUI(true);
                }
            };
            reader.onerror = (e) => {
                console.error("FileReader error:", e);
                showError("Error reading the selected file.");
                resetUI(true);
            };
            reader.readAsDataURL(file);
        } else if (sourceType === 'example') {
            // Preview already shown
        }
        if (predictButton) predictButton.disabled = false;
        hideError();
        hideResult();
        hideLoading();
        hideTemporaryLoading();
    }

    async function handlePrediction() {
        if (!currentFile) {
            showError("No image is ready for recognition. Please select one.");
            console.error("Prediction attempt without a valid file.");
            return;
        }
        if (!predictButton || !loadingIndicator || !resultDisplay || !errorDisplay) {
             console.error("One or more required UI elements not found for prediction.");
             showError("UI error. Cannot proceed with prediction.");
             return;
        }
        console.log(`Sending file "${currentFile.name}" (${currentFile.type}) for prediction to ${API_PREDICT_ENDPOINT}`);
        showLoading();
        hideResult();
        hideError();
        predictButton.disabled = true;
        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(API_PREDICT_ENDPOINT, {
                method: 'POST',
                body: formData,
            });

            // --- START: DETAILED DEBUG LOGGING FOR RESPONSE ---
            console.log("[DEBUG] Fetch Response Status:", response.status, response.statusText);

            let data;
            let responseText = ''; // Declare here to access in catch
            try {
                responseText = await response.text(); // Get raw text first
                console.log("[DEBUG] Raw Response Text:", responseText);
                data = JSON.parse(responseText); // THEN parse
                console.log("[DEBUG] Parsed JSON Data:", JSON.stringify(data, null, 2)); // Log parsed data
            } catch (jsonError) {
                console.error("[DEBUG] JSON Parsing Error:", jsonError);
                console.error("[DEBUG] Raw text that failed to parse:", responseText); // Log raw text on error
                throw new Error(`Server returned invalid JSON (Status: ${response.status})`);
            }

            // Check HTTP status *after* attempting to parse
            if (!response.ok) {
                const errorMsg = data?.error || `Request failed: ${response.status} ${response.statusText}`;
                console.error("[DEBUG] HTTP request failed:", errorMsg);
                throw new Error(errorMsg);
            }

            // Check for application-level error in the parsed data
            if (data.error) {
                 console.error("[DEBUG] Application error from backend:", data.error);
                 throw new Error(data.error);
            }

            // Check if prediction data exists AND IS THE CORRECT TYPE
            if (data.prediction !== undefined && typeof data.prediction === 'string' &&
                data.probability !== undefined && typeof data.probability === 'number')
            {
                console.log("[DEBUG] Prediction data looks valid. Calling showResult...");
                showResult(data.prediction, data.probability); // Call the UI update function
            } else {
                 console.error("[DEBUG] Received successful response, but prediction/probability data is missing or invalid type.");
                 console.error("[DEBUG] Data received:", data); // Log the problematic data
                 throw new Error("Received unexpected response format from the server.");
            }
            // --- END: DETAILED DEBUG LOGGING FOR RESPONSE ---

        } catch (error) {
            console.error("Prediction request failed:", error);
            showError(`Prediction failed: ${error.message || "Check console for details."}`);
        } finally {
            hideLoading();
            if (predictButton) predictButton.disabled = !currentFile;
        }
    }

    // --- UI Update Functions ---

    function showLoading() { loadingIndicator?.classList.remove('hidden'); }
    function hideLoading() { loadingIndicator?.classList.add('hidden'); }

    function showTemporaryLoading(message = "Loading...") {
        if (fileNameSpan) {
            fileNameSpan.textContent = message;
            fileNameSpan.classList.add('text-primary', 'animate-pulse');
        }
    }
    function hideTemporaryLoading() {
         if (fileNameSpan) {
            fileNameSpan.textContent = currentFile ? currentFile.name : 'No file chosen';
            fileNameSpan.classList.remove('text-primary', 'animate-pulse');
            fileNameSpan.title = currentFile ? currentFile.name : '';
        }
    }

    function showResult(prediction, probability) {
        // --- START: DETAILED DEBUG LOGGING FOR showResult ---
        console.log(`--- ENTERING showResult ---`);
        console.log(`[DEBUG] Prediction Param: "${prediction}" (Type: ${typeof prediction})`);
        console.log(`[DEBUG] Probability Param: ${probability} (Type: ${typeof probability})`);
        console.log("[DEBUG] Element References:", { resultDisplay, predictionText, probabilityText });
        console.log("[DEBUG] Current predictionText content:", predictionText?.textContent);
        console.log("[DEBUG] Current probabilityText content:", probabilityText?.textContent);
        console.log("[DEBUG] Is resultDisplay hidden?", resultDisplay?.classList.contains('hidden'));
        // --- END: DETAILED DEBUG LOGGING ---

        if (!resultDisplay || !predictionText || !probabilityText) {
            console.error("showResult: One or more result display elements not found!");
            return;
        }
        const probPercent = (probability * 100).toFixed(1);
        console.log("[DEBUG] Setting predictionText content to:", prediction);
        predictionText.textContent = prediction;
        console.log("[DEBUG] Setting probabilityText content to:", `(Probability: ${probPercent}%)`);
        probabilityText.textContent = `(Probability: ${probPercent}%)`;
        console.log("[DEBUG] Removing 'hidden' from resultDisplay");
        resultDisplay.classList.remove('hidden');
        console.log("[DEBUG] Is resultDisplay hidden AFTER removing class?", resultDisplay?.classList.contains('hidden'));
        console.log(`--- EXITING showResult ---`);
    }

    function hideResult() {
        if (!resultDisplay) return;
        resultDisplay.classList.add('hidden');
        if (predictionText) predictionText.textContent = '';
        if (probabilityText) probabilityText.textContent = '';
    }

    function showError(message) {
        if (!errorDisplay || !errorText) {
            console.error("Cannot show error - error display elements not found.");
            alert(`Error: ${message}`);
            return;
        }
        errorText.textContent = message;
        errorDisplay.classList.remove('hidden');
    }

    function hideError() {
        if (!errorDisplay) return;
        errorDisplay.classList.add('hidden');
        if (errorText) errorText.textContent = '';
    }

    function showImagePreview(imageSrc) {
        if (!imagePreview || !imagePreviewContainer) {
            console.warn("Image preview elements not found.");
            return;
        }
        if (currentObjectURL) {
            URL.revokeObjectURL(currentObjectURL);
            console.log("Revoked previous Object URL:", currentObjectURL);
            currentObjectURL = null;
        }
        imagePreview.src = imageSrc;
        if (imageSrc && imageSrc.startsWith('blob:')) {
            currentObjectURL = imageSrc;
            console.log("Set new Object URL for preview:", currentObjectURL);
        }
        imagePreviewContainer.classList.remove('hidden');
    }

    function hideImagePreview() {
         if (!imagePreview || !imagePreviewContainer) return;
        imagePreview.src = '#';
        imagePreviewContainer.classList.add('hidden');
        if (currentObjectURL) {
            URL.revokeObjectURL(currentObjectURL);
            console.log("Revoked Object URL on hide:", currentObjectURL);
            currentObjectURL = null;
        }
    }

    function displayFileName(name) {
         if (!fileNameSpan) return;
        const displayName = name || 'No file chosen';
        fileNameSpan.textContent = displayName;
        fileNameSpan.title = name || '';
    }

    function resetUI(clearFileInput = true) {
        console.log(`Resetting UI. Clear input: ${clearFileInput}`);
        currentFile = null;
        if (clearFileInput && imageUploadInput) {
            imageUploadInput.value = '';
        }
        displayFileName(null);
        hideImagePreview();
        if (predictButton) predictButton.disabled = true;
        hideLoading();
        hideTemporaryLoading();
        hideResult();
        hideError();
    }

}); // End DOMContentLoaded Listener