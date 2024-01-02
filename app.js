document.addEventListener('DOMContentLoaded', async function () {
    const imageInput = document.getElementById('image-input');
    const outputContainer = document.getElementById('output-container');

    // Load the DeepLab model
    const model = await tf.loadLayersModel('https://tfhub.dev/tensorflow/tfjs-model/deeplab/pascal/1/default/1');

    // Event listener for file input change
    imageInput.addEventListener('change', handleFileSelect);

    // Handle file selection
    async function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            const image = await loadImage(file);
            outputContainer.innerHTML = ''; // Clear previous results
            outputContainer.appendChild(image);
            detectRoad(image);
        }
    }

    // Load an image and create an HTML image element
    async function loadImage(file) {
        return new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = function () {
                const image = new Image();
                image.src = reader.result;
                image.onload = () => resolve(image);
            };
            reader.readAsDataURL(file);
        });
    }

    // Detect roads in the input image
    async function detectRoad(image) {
        const inputTensor = tf.browser.fromPixels(image);
        const resizedImage = tf.image.resizeBilinear(inputTensor, [513, 513]);
        const input = tf.expandDims(resizedImage.div(255), 0);

        // Run the model
        const segmentation = await model.predict(input);

        // Render the segmentation mask
        const segmentationData = await segmentation.data();
        const [height, width] = segmentation.shape.slice(1);
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = width;
        maskCanvas.height = height;
        const maskCtx = maskCanvas.getContext('2d');
        const maskImageData = maskCtx.createImageData(width, height);

        for (let i = 0; i < width * height; i++) {
            const j = i * 4;
            const classIndex = segmentationData[i];
            maskImageData.data[j] = 0;
            maskImageData.data[j + 1] = classIndex === 15 ? 255 : 0; // Road class index is 15
            maskImageData.data[j + 2] = 0;
            maskImageData.data[j + 3] = 255;
        }

        maskCtx.putImageData(maskImageData, 0, 0);
        outputContainer.appendChild(maskCanvas);
    }
});
