let modelMask;
let maskArray      = new Array();
let detectionArray = new Array();

let avgMaskResult      = 0.1;
let avgDetectionResult = 0.1;

let maskDetected = false;
let oldMaskDetected;

(async function() {
    modelMask = await tf.loadLayersModel("models/modelMask.json");
})();

const video = document.getElementById('video')

Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri('models')
]).then(startVideo)

function startVideo() {
    navigator.mediaDevices.getUserMedia({
        video: {}
    }).then((stream) => {
        video.srcObject = stream;
    }, (err) => console.error(err));
}

video.addEventListener('play', () => {
    const canvas = faceapi.createCanvasFromMedia(video)
    document.getElementById('magic').append(canvas);
    const displaySize = {
        width: video.width,
        height: video.height
    }
    faceapi.matchDimensions(canvas, displaySize)
    setInterval(async () => {
        const detection = await faceapi.detectSingleFace(video, new faceapi.SsdMobilenetv1Options())

        if (detection) {
            avgDetectionResult = average(detection['score'], detectionArray);

            const resizedDetection = faceapi.resizeResults(detection, displaySize)
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
            faceapi.draw.drawDetections(canvas, resizedDetection)

            let left = parseInt(detection['box']['topLeft']['x']);
            let top = parseInt(detection['box']['topLeft']['y']);
            let right = parseInt(detection['box']['bottomRight']['x']);
            let bottom = parseInt(detection['box']['bottomRight']['y']);

            let facecanvas = document.getElementById('facecanvas');
            let ctx = facecanvas.getContext('2d');

            ctx.drawImage(video,
                left - 10, top - 10, // add small padding
                right - left + 20, bottom - top + 20, // add small padding
                0, 0, // add small padding
                224, 224); // scale to 224px
            checkMask();
            if (maskDetected != oldMaskDetected) {
                changeContent();
                oldMaskDetected = maskDetected;
            }
        } else if (avgDetectionResult > 0.2) {
            // try to eliminate false (non detections)
            avgDetectionResult = average(0, detectionArray);
        } else {
            if (maskDetected) {
                oldMaskDetected = false
            } else {
                oldMaskDetected = true
            }
            document.getElementById("data-container").style.backgroundColor = "#3298dc";
            document.getElementById("data-header-title").innerHTML = "ATTENTION!";
            document.getElementById("data-footer-title").innerHTML = "RAPPROCHEZ-VOUS DE L'ECRAN";
        }
    }, 100)

})

async function checkMask() {
    const faceImage = document.getElementById('facecanvas');

    let tensor = tf.browser.fromPixels(faceImage);
    const offset = tf.scalar(127.5);
    const normalized = tensor.sub(offset).div(offset);
    const batched = normalized.reshape([1, 224, 224, 3]);

    let predictions = await modelMask.predict(batched).data();

    avgMaskResult = average(predictions[0], maskArray);

    if (avgMaskResult < 0.5) {
        maskDetected = false
    } else {
        maskDetected = true
    }
}

function changeContent () {
    if (avgMaskResult < 0.5) {
        document.getElementById("data-container").style.backgroundColor = "#f14668";
        document.getElementById("data-header-title").innerHTML = "STOP!";
        document.getElementById("data-footer-title").innerHTML = "PORTER UN MASQUE EST OBLIGATOIRE";
    } else {
        document.getElementById("data-container").style.backgroundColor = "#48c774";
        document.getElementById("data-header-title").innerHTML = "MERCI!";
        document.getElementById("data-footer-title").innerHTML = "MERCI POUR PORTER VOTRE MASQUE!";
    }
}

function average(value, array) {
    array.unshift(value);
    array = array.slice(0, 15);

    var total = 0;
    for(var i = 0; i < array.length; i++) {
        total += array[i];
    }
    var avg = total / array.length;
    return avg;
}
