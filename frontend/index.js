
let model;
const defaultParams = {
    flipHorizontal: false,
    outputStride: 16,
    imageScaleFactor: 1,
    maxNumBoxes: 20,
    iouThreshold: 0.2,
    scoreThreshold: 0.6,
    modelType: "ssd320fpnlite",
    modelSize: "large",
    bboxLineWidth: "2",
    fontSize: 17,
};

async function init() {
   
    model =  await handTrack.load(defaultParams);
}

function drawAndSave(context,x,y,width,height){
    context.beginPath();
    context.rect(x, y, width, height);
    context.lineWidth = 2;
    context.strokeStyle = 'red';
    context.fillStyle = 'transparent';
    context.stroke();
    context.closePath();
    
    const newCanvas = document.createElement('canvas');
    const newContext = newCanvas.getContext('2d');

    
    newCanvas.width = width;
    newCanvas.height = height;

    const imageData = context.getImageData(x, y, width, height);

    newContext.putImageData(imageData, 0, 0);

    const extractedImage = new Image();
    extractedImage.src = newCanvas.toDataURL('image/png');

    document.body.appendChild(extractedImage);
}

async function predict(img){
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const predictions = await model.detect(img); 
    console.log(predictions);
    context.drawImage(img, 0, 0, img.width,img.height);
    for(let i =0;i<predictions.length;i++){

        if(predictions[i].label!=="face"){
            let [x, y, width, height] = predictions[i].bbox;
            x = x-20
            y = y - 20
            width = width + 50
            height = height + 20
            drawAndSave(context,x,y,width,height)
        }
    }
    console.log(predictions);
    
}


window.onload = function() {
    init();
    const video = document.getElementById('video');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
            };
        })
        .catch((error) => {
            console.error('Error accessing camera:', error);
        });
};


const video = document.getElementById('video');
let frameNumber = 0;

video.addEventListener('loadedmetadata', function () {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 640
    canvas.height = 480
    video.addEventListener('play', function () {
        captureFrame();
    });

    function captureFrame() {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        console.log(canvas.width, canvas.height);
        const img = new Image();
        img.src = canvas.toDataURL('image/png');
        predict(img)
        if (frameNumber < video.duration * video.playbackRate) {
            requestAnimationFrame(captureFrame);
        }
    }

    video.play();
});