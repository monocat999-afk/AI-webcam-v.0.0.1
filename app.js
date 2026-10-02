// Wait for HTML to fully load before running scripts
document.addEventListener('DOMContentLoaded', () => {

    // DOM Elements
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    const statusBanner = document.getElementById('status-banner');
    const classesList = document.getElementById('classes-list');
    const predictionText = document.getElementById('prediction-text');
    
    // ML Global Variables
    let knn, mobilenetModel, cocoModel;
    let isPredicting = false;
    let captureInterval = null;
    let classes = [];
    
    const CONFIDENCE_THRESHOLD = 0.70; 
    const EMBEDDING_SIZE = 1024;       

    // UI Feedback
    function showStatus(msg, type = 'success') {
        statusBanner.className = 'px-4 py-2 text-sm font-bold text-center transition-all rounded-none border-y';
        if (type === 'error') {
            statusBanner.classList.add('bg-red-900/80', 'border-red-500', 'text-red-200');
        } else if (type === 'success') {
            statusBanner.classList.add('bg-emerald-900/80', 'border-emerald-500', 'text-emerald-200');
        } else {
            statusBanner.classList.add('bg-cyan-900/80', 'border-cyan-500', 'text-cyan-200');
        }
        statusBanner.innerText = msg;
        statusBanner.classList.remove('hidden');
        if (type !== 'error') setTimeout(() => statusBanner.classList.add('hidden'), 4000);
    }

    async function initHardwareBackend(backendName) {
        try {
            await tf.setBackend(backendName);
            await tf.ready();
            console.log(`Backend Active: ${tf.getBackend()}`);
        } catch (err) {
            await tf.setBackend('cpu');
        }
    }

    async function setupCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve(video);
            };
        });
    }

    async function initApp() {
        try {
            await initHardwareBackend('webgl');
            document.getElementById('backend-select').value = tf.getBackend();
            await setupCamera();
            
            knn = knnClassifier.create();
            mobilenetModel = await mobilenet.load(); 
            cocoModel = await cocoSsd.load();        

            document.getElementById('camera-loading').remove();
            showStatus("System Ready: Neural Engines Online", 'success');
            
            isPredicting = true;
            predictLoop();
        } catch (err) {
            showStatus(`Error: ${err.message}`, 'error');
        }
    }

    function addClass(className, restoreId = null, restoreCount = 0) {
        const name = className.trim();
        if (!name) return;
        const classId = restoreId !== null ? restoreId : classes.length;
        
        if (restoreId === null) classes.push({ id: classId, name: name, count: 0 });

        const item = document.createElement('div');
        item.className = "flex items-center justify-between bg-slate-800 p-3 rounded-lg border border-slate-700 shadow-sm";
        item.innerHTML = `
            <div class="flex flex-col">
                <span class="text-sm font-bold text-slate-200">${name}</span>
                <span class="text-xs text-slate-400 font-mono">Samples: <span id="count-${classId}" class="text-cyan-400">${restoreCount}</span></span>
            </div>
            <button id="btn-${classId}" class="bg-cyan-600 text-white text-xs px-4 py-2 rounded-md font-bold transition-all">
                Hold to Capture
            </button>
        `;
        classesList.appendChild(item);

        const captureBtn = item.querySelector(`#btn-${classId}`);
        
        const startCapture = (e) => {
            if(e) e.preventDefault();
            captureBtn.classList.replace('bg-cyan-600', 'bg-emerald-500');
            captureBtn.innerText = "Capturing...";
            captureFrame(classId); 
            captureInterval = setInterval(() => captureFrame(classId), 100); 
        };
        
        const stopCapture = (e) => {
            if(e) e.preventDefault();
            if (captureInterval) clearInterval(captureInterval);
            captureBtn.classList.replace('bg-emerald-500', 'bg-cyan-600');
            captureBtn.innerText = "Hold to Capture";
        };

        captureBtn.addEventListener('mousedown', startCapture);
        captureBtn.addEventListener('mouseup', stopCapture);
        captureBtn.addEventListener('mouseleave', stopCapture);
        captureBtn.addEventListener('touchstart', startCapture, {passive: false});
        captureBtn.addEventListener('touchend', stopCapture);
    }

    function captureFrame(classId) {
        if (!mobilenetModel || video.readyState !== 4) return;
        const logits = mobilenetModel.infer(video, true);
        knn.addExample(logits, classId);
        logits.dispose(); // RAM Fix
        
        const targetClass = classes.find(c => c.id === classId);
        targetClass.count++;
        document.getElementById(`count-${classId}`).innerText = targetClass.count;
    }

    async function predictLoop() {
        while (isPredicting) {
            if (video.readyState === 4 && video.videoWidth > 0) {
                if(canvas.width !== video.videoWidth) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                }

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (cocoModel) {
                    const predictions = await cocoModel.detect(video);
                    predictions.forEach(pred => {
                        if (pred.score > 0.5) {
                            const [x, y, width, height] = pred.bbox;
                            ctx.save();
                            ctx.globalCompositeOperation = 'difference';
                            ctx.strokeStyle = '#ffffff'; 
                            ctx.lineWidth = 3;
                            ctx.strokeRect(x, y, width, height);

                            ctx.fillStyle = '#ffffff';
                            ctx.font = 'bold 16px system-ui';
                            ctx.fillText(`${pred.class.toUpperCase()} ${Math.round(pred.score * 100)}%`, x + 5, y > 20 ? y - 8 : y + 20);
                            ctx.restore();
                        }
                    });
                }

                if (knn && knn.getNumClasses() > 0) {
                    const logits = mobilenetModel.infer(video, true);
                    const res = await knn.predictClass(logits);
                    logits.dispose();
                    
                    const confidence = res.confidences[res.label];
                    if (confidence < CONFIDENCE_THRESHOLD) {
                        predictionText.innerText = "Unknown / Untrained Object";
                        predictionText.className = "text-2xl font-bold text-amber-500 truncate";
                    } else {
                        const matchedClass = classes.find(c => c.id == res.label);
                        predictionText.innerText = `${matchedClass ? matchedClass.name : 'Class ' + res.label} (${Math.round(confidence * 100)}%)`;
                        predictionText.className = "text-2xl font-bold text-emerald-400 truncate";
                    }
                }
            }
            await tf.nextFrame();
        }
    }

    // Events
    document.getElementById('add-class-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const input = document.getElementById('new-class-name');
        addClass(input.value);
        input.value = '';
    });

    document.getElementById('backend-select').addEventListener('change', async (e) => {
        isPredicting = false; 
        await initHardwareBackend(e.target.value);
        isPredicting = true;
        predictLoop();
    });

    document.getElementById('export-btn').addEventListener('click', () => {
        if(!knn || knn.getNumClasses() === 0) return showStatus("No data to save!", "error");
        
        const dataset = knn.getClassifierDataset();
        let datasetObj = {};
        Object.keys(dataset).forEach(key => {
            datasetObj[key] = Array.from(dataset[key].dataSync());
        });
        
        const blob = new Blob([JSON.stringify({ dataset: datasetObj, classes: classes })], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'vision_model.json';
        a.click();
        URL.revokeObjectURL(a.href);
    });

    document.getElementById('import-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            const data = JSON.parse(evt.target.result);
            let tensorObj = {};
            Object.keys(data.dataset).forEach(key => {
                tensorObj[key] = tf.tensor2d(data.dataset[key], [data.dataset[key].length / EMBEDDING_SIZE, EMBEDDING_SIZE]);
            });
            knn.setClassifierDataset(tensorObj);
            
            if(data.classes) {
                classes = data.classes;
                classesList.innerHTML = ''; 
                classes.forEach(c => addClass(c.name, c.id, c.count));
            }
            showStatus("Model Restored Successfully!", "success");
        };
        reader.readAsText(file);
        e.target.value = '';
    });

    // Start App
    initApp();
});