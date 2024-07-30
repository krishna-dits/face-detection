const express = require('express');
const multer = require('multer');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const path = require('path');
const fs = require('fs');
const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const upload = multer({ dest: 'uploads/' });

const PORT = process.env.PORT || 3000;

// Load models
Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models')),
    faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, 'models')),
    faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'))
]).then(() => {
    console.log('Models loaded');
});

// Helper function to get face descriptor
const getFaceDescriptor = async (imagePath) => {
    const img = await canvas.loadImage(imagePath);
    const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
    return detection ? detection.descriptor : null;
};

app.get('/', (req, res) => {
    res.send("face api");
})

// Route to handle face comparison
app.post('/compare', upload.fields([{ name: 'image1', maxCount: 1 }, { name: 'image2', maxCount: 1 }]), async (req, res) => {
    try {
        const files = req.files;
        if (!files || !files.image1 || !files.image2) {
            return res.status(400).json({ error: 'Please upload two images' });
        }

        const descriptor1 = await getFaceDescriptor(files.image1[0].path);
        const descriptor2 = await getFaceDescriptor(files.image2[0].path);

        if (!descriptor1 || !descriptor2) {
            return res.status(400).json({ error: 'Could not detect faces in both images' });
        }

        const distance = faceapi.euclideanDistance(descriptor1, descriptor2);
        const isMatch = distance < 0.5;

        res.json({ distance, isMatch });

        // Clean up uploaded files
        fs.unlinkSync(files.image1[0].path);
        fs.unlinkSync(files.image2[0].path);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
