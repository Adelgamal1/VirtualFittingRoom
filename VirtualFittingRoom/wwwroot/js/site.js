(() => {
    document.querySelectorAll("[data-password-toggle]").forEach(toggle => {
        const container = toggle.closest(".password-field");
        const input = container?.querySelector(".password-input");
        const icon = toggle.querySelector("i");

        if (!input || !icon) {
            return;
        }

        toggle.addEventListener("click", () => {
            const shouldShow = input.type === "password";
            input.type = shouldShow ? "text" : "password";
            icon.className = shouldShow ? "bi bi-eye-slash" : "bi bi-eye";
            toggle.setAttribute("aria-label", shouldShow ? "Hide password" : "Show password");
        });
    });

    const loadScriptOnce = (src) => new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }

        const script = document.createElement("script");
        script.src = src;
        script.async = true;
        script.onload = resolve;
        script.onerror = () => reject(new Error(`Could not load ${src}`));
        document.head.appendChild(script);
    });

    const poseConnections = [
        [11, 12],
        [11, 13],
        [13, 15],
        [12, 14],
        [14, 16],
        [11, 23],
        [12, 24],
        [23, 24]
    ];
    let staticPoseModel = null;
    let staticPosePromise = null;
    let staticPoseResolver = null;

    const serializePoseLandmarks = (landmarks, width, height) => JSON.stringify({
        width,
        height,
        landmarks: (landmarks || []).map(landmark => ({
            x: Number((landmark.x ?? 0).toFixed(6)),
            y: Number((landmark.y ?? 0).toFixed(6)),
            z: Number((landmark.z ?? 0).toFixed(6)),
            visibility: Number((landmark.visibility ?? 1).toFixed(4))
        }))
    });

    const isStaticPoseUsable = (landmarks) => {
        if (!landmarks?.length) {
            return false;
        }

        const visible = (index, min = 0.25) => (landmarks[index]?.visibility ?? 0) >= min;
        if (![11, 12, 23, 24].every(index => visible(index))) {
            return false;
        }

        const leftShoulder = landmarks[11];
        const rightShoulder = landmarks[12];
        const leftHip = landmarks[23];
        const rightHip = landmarks[24];
        const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        const hipY = (leftHip.y + rightHip.y) / 2;
        const shoulderWidth = Math.abs(rightShoulder.x - leftShoulder.x);
        const hipWidth = Math.abs(rightHip.x - leftHip.x);

        return shoulderY >= 0.12 &&
            shoulderY <= 0.42 &&
            hipY >= 0.40 &&
            hipY <= 0.86 &&
            hipY - shoulderY >= 0.18 &&
            shoulderWidth >= 0.10 &&
            shoulderWidth <= 0.55 &&
            hipWidth >= 0.06 &&
            hipWidth <= 0.48;
    };

    const clearPoseOverlay = (canvas, input) => {
        if (input) {
            input.value = "";
        }

        if (!canvas) {
            return;
        }

        const context = canvas.getContext("2d");
        context?.clearRect(0, 0, canvas.width, canvas.height);
        canvas.classList.add("d-none");
    };

    const initializeStaticPoseModel = async () => {
        if (staticPoseModel) {
            return staticPoseModel;
        }

        if (!staticPosePromise) {
            staticPosePromise = loadScriptOnce("https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js")
                .then(() => {
                    staticPoseModel = new window.Pose({
                        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
                    });
                    staticPoseModel.setOptions({
                        modelComplexity: 1,
                        smoothLandmarks: true,
                        enableSegmentation: false,
                        selfieMode: false,
                        minDetectionConfidence: 0.55,
                        minTrackingConfidence: 0.55
                    });
                    staticPoseModel.onResults(results => {
                        const resolver = staticPoseResolver;
                        staticPoseResolver = null;
                        resolver?.(results);
                    });
                    return staticPoseModel;
                });
        }

        return staticPosePromise;
    };

    const loadImageElement = (src) => new Promise((resolve, reject) => {
        if (!src) {
            reject(new Error("Image source is empty."));
            return;
        }

        const image = new Image();
        if (/^https?:\/\//i.test(src)) {
            image.crossOrigin = "anonymous";
        }
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = src;
    });

    const runStaticPoseDetection = async (image) => {
        const model = await initializeStaticPoseModel();
        return await new Promise(resolve => {
            let finished = false;
            const finish = (results) => {
                if (finished) {
                    return;
                }

                finished = true;
                resolve(results);
            };

            staticPoseResolver = finish;
            window.setTimeout(() => finish(null), 3800);
            model.send({ image }).catch(() => finish(null));
        });
    };

    const drawPoseOverlay = (preview, canvas, landmarks) => {
        if (!preview || !canvas || !landmarks?.length) {
            clearPoseOverlay(canvas, null);
            return;
        }

        const width = Math.max(1, Math.round(preview.clientWidth || preview.width || 1));
        const height = Math.max(1, Math.round(preview.clientHeight || preview.height || 1));
        canvas.width = width;
        canvas.height = height;

        const naturalWidth = preview.naturalWidth || width;
        const naturalHeight = preview.naturalHeight || height;
        const scale = Math.min(width / naturalWidth, height / naturalHeight);
        const drawnWidth = naturalWidth * scale;
        const drawnHeight = naturalHeight * scale;
        const offsetX = (width - drawnWidth) / 2;
        const offsetY = (height - drawnHeight) / 2;
        const map = landmark => ({
            x: offsetX + (landmark.x * drawnWidth),
            y: offsetY + (landmark.y * drawnHeight),
            visibility: landmark.visibility ?? 1
        });
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, width, height);
        context.lineWidth = Math.max(2, width * 0.006);
        context.lineCap = "round";
        context.strokeStyle = "rgba(34, 211, 238, 0.92)";
        context.fillStyle = "rgba(255, 255, 255, 0.95)";

        poseConnections.forEach(([from, to]) => {
            const a = landmarks[from];
            const b = landmarks[to];
            if (!a || !b || (a.visibility ?? 1) < 0.2 || (b.visibility ?? 1) < 0.2) {
                return;
            }

            const start = map(a);
            const end = map(b);
            context.beginPath();
            context.moveTo(start.x, start.y);
            context.lineTo(end.x, end.y);
            context.stroke();
        });

        [0, 11, 12, 13, 14, 15, 16, 23, 24].forEach(index => {
            const landmark = landmarks[index];
            if (!landmark || (landmark.visibility ?? 1) < 0.2) {
                return;
            }

            const point = map(landmark);
            context.beginPath();
            context.arc(point.x, point.y, Math.max(3, width * 0.008), 0, Math.PI * 2);
            context.fill();
        });

        canvas.classList.remove("d-none");
    };

    const detectAndStorePose = async (src, preview, canvas, input) => {
        clearPoseOverlay(canvas, input);

        try {
            const image = await loadImageElement(src);
            const results = await runStaticPoseDetection(image);
            const landmarks = results?.poseLandmarks || [];
            if (!isStaticPoseUsable(landmarks)) {
                clearPoseOverlay(canvas, input);
                return "";
            }

            const payload = serializePoseLandmarks(landmarks, image.naturalWidth || image.width, image.naturalHeight || image.height);
            if (input) {
                input.value = payload;
            }
            drawPoseOverlay(preview, canvas, landmarks);
            return payload;
        } catch {
            clearPoseOverlay(canvas, input);
            return "";
        }
    };

    const normalizeCatalogValue = (value) => (value || "")
        .trim()
        .toLowerCase()
        .replace(/[_\s]+/g, "-");

    const normalizeTargetBody = (value) => {
        const normalized = normalizeCatalogValue(value);
        return {
            man: "male",
            men: "male",
            woman: "female",
            women: "female",
            kid: "child",
            kids: "child",
            children: "child"
        }[normalized] || normalized;
    };

    const categoryAreaMap = {
        "pants": "lower",
        "trousers": "lower",
        "jeans": "lower",
        "short": "lower",
        "shorts": "lower",
        "dress": "overall",
        "jumpsuit": "overall",
        "overall": "overall",
        "overalls": "overall",
        "romper": "overall",
        "salopette": "overall",
        "salopeit": "overall",
        "سالوبيت": "overall",
        "abaya": "overall",
        "abayas": "overall",
        "عباية": "overall",
        "عبايات": "overall",
        "galabeya": "overall",
        "galabiya": "overall",
        "jellabiya": "overall",
        "t-shirt": "upper",
        "tee": "upper",
        "tshirt": "upper",
        "jersey": "upper",
        "sports-shirt": "upper",
        "sport-shirt": "upper",
        "hockey-jersey": "upper",
        "football-jersey": "upper",
        "basketball-jersey": "upper",
        "tank-top": "upper",
        "tanktop": "upper",
        "shirt": "upper",
        "chemise": "upper",
        "blouse": "upper",
        "hoodie": "upper",
        "jacket": "upper"
    };

    const categoryToArea = (value, option = null) =>
        option?.dataset.area || categoryAreaMap[normalizeCatalogValue(value)] || "";

    const isOptionAllowedForTarget = (option, targetBody) => {
        if (!option?.value || !targetBody) {
            return true;
        }

        const audiences = (option.dataset.audience || "")
            .split(/\s+/)
            .map(normalizeTargetBody)
            .filter(Boolean);

        return audiences.length === 0 || audiences.includes(targetBody);
    };

    const syncCategoryCatalog = (targetSelect, categorySelect, areaSelect) => {
        if (!categorySelect) {
            return "";
        }

        const targetBody = normalizeTargetBody(targetSelect?.value);
        let selectedOptionAllowed = true;
        let firstAllowedOption = null;
        let preferredOption = null;
        const preferredByTarget = {
            male: "T-Shirt",
            female: "Blouse",
            child: "T-Shirt"
        }[targetBody];

        Array.from(categorySelect.options).forEach(option => {
            if (!option.value) {
                return;
            }

            const allowed = isOptionAllowedForTarget(option, targetBody);
            option.hidden = !allowed;
            option.disabled = !allowed;

            if (allowed && !firstAllowedOption) {
                firstAllowedOption = option;
            }

            if (allowed && preferredByTarget && option.value === preferredByTarget) {
                preferredOption = option;
            }

            if (option.selected && !allowed) {
                selectedOptionAllowed = false;
            }
        });

        if ((!categorySelect.value || !selectedOptionAllowed) && (preferredOption || firstAllowedOption)) {
            categorySelect.value = (preferredOption || firstAllowedOption).value;
        }

        const selected = categorySelect.selectedOptions?.[0] || null;
        const mappedArea = categoryToArea(categorySelect.value, selected);
        if (mappedArea && areaSelect) {
            areaSelect.value = mappedArea;
            areaSelect.dataset.autoArea = mappedArea;

            Array.from(areaSelect.options).forEach(option => {
                const isAreaOption = Boolean(option.value);
                option.hidden = isAreaOption && option.value !== mappedArea;
                option.disabled = isAreaOption && option.value !== mappedArea;
            });
        }

        return mappedArea;
    };

    const form = document.querySelector("[data-tryon-form]");

    if (form) {
        const isPublicLive = form.dataset.publicLive === "true";
        const initialClothingUrl = (form.dataset.initialClothingUrl || "").trim();
        const livePreviewEnabled = form.dataset.livePreview === "true";
        const shouldAutoStartCamera = form.dataset.autoCamera === "true";
        const sourceButtons = form.querySelectorAll("[data-source-trigger]");
        const cameraSection = form.querySelector("[data-camera-section]");
        const posePreviewShell = form.querySelector("[data-pose-preview-shell]");
        const poseUploadSection = form.querySelector("[data-pose-upload-section]");
        const poseUrlSection = form.querySelector("[data-pose-url-section]");
        const cameraStatus = form.querySelector("[data-camera-status]");
        const video = form.querySelector("[data-camera-feed]");
        const canvas = form.querySelector("[data-camera-canvas]");
        const liveOverlayCanvas = form.querySelector("[data-live-overlay-canvas]");
        const liveGarmentOverlay = form.querySelector("[data-live-garment-overlay]");
        const fileCameraPanel = form.querySelector("[data-file-camera-panel]");
        const fileCameraFeed = form.querySelector("[data-file-camera-feed]");
        const fileCameraCanvas = form.querySelector("[data-file-camera-canvas]");
        const fileCameraOpen = form.querySelector("[data-file-camera-open]");
        const fileCameraCapture = form.querySelector("[data-file-camera-capture]");
        const fileCameraClose = form.querySelector("[data-file-camera-close]");
        const imageData = form.querySelector("[data-image-data]");
        const clothingImageData = form.querySelector("[data-clothing-image-data]");
        const poseLandmarksData = form.querySelector("[data-pose-landmarks]");
        const poseUpload = form.querySelector("[data-pose-upload]");
        const clothingUpload = form.querySelector("[data-clothing-upload]");
        const poseUploadTriggers = form.querySelectorAll("[data-pose-upload-trigger]");
        const clothingUploadTriggers = form.querySelectorAll("[data-clothing-upload-trigger]");
        const clothingLinkTriggers = form.querySelectorAll("[data-clothing-link-trigger]");
        const poseLinkTriggers = form.querySelectorAll("[data-pose-link-trigger]");
        const liveClothingUploadTriggers = form.querySelectorAll("[data-live-clothing-upload-trigger]");
        const liveClothingLinkTriggers = form.querySelectorAll("[data-live-clothing-link-trigger]");
        const liveAiFitTrigger = form.querySelector("[data-live-ai-fit]");
        const driveOpenTriggers = form.querySelectorAll("[data-drive-open]");
        const storeOpenTriggers = form.querySelectorAll("[data-store-open]");
        const clothingLinkPanel = form.querySelector("[data-clothing-link-panel]");
        const poseLinkPanel = form.querySelector("[data-pose-link-panel]");
        const liveClothingLinkPanel = form.querySelector("[data-live-clothing-link-panel]");
        const poseUrl = form.querySelector("[data-pose-url]");
        const clothingUrl = form.querySelector("[data-clothing-url]");
        const liveClothingUrl = form.querySelector("[data-live-clothing-url]");
        const posePreview = form.querySelector("[data-pose-preview]");
        const poseOverlayCanvas = form.querySelector("[data-pose-overlay]");
        const poseEmpty = form.querySelector("[data-pose-empty]");
        const clothingPreview = form.querySelector("[data-clothing-preview]");
        const clothingEmpty = form.querySelector("[data-clothing-empty]");
        const targetBody = form.querySelector("[data-target-body]");
        const clothingCategory = form.querySelector("[data-clothing-category]");
        const garmentArea = form.querySelector("[data-garment-area]");
        const garmentView = form.querySelector("[data-garment-view]");
        const submitRow = form.querySelector("[data-submit-row]");
        const submitButton = submitRow?.querySelector("[type='submit']") || form.querySelector("[type='submit']");
        const poseDebugToggle = form.querySelector("[data-pose-debug-toggle]");
        const fittingDebugOutput = form.querySelector("[data-fitting-debug-output]");
        let stream = null;
        let fileCameraStream = null;
        let poseResizePromise = Promise.resolve();
        let clothingResizePromise = Promise.resolve();
        let isSubmittingAfterResize = false;
        let liveGarmentImage = null;
        let liveGarmentUrl = "";
        let liveOverlayFrame = 0;
        let bodyTrackingPromise = null;
        let poseModel = null;
        let segmentationModel = null;
        let latestPoseLandmarks = null;
        let latestSegmentationMask = null;
        let trackingLoopActive = false;
        let isTrackingFrame = false;
        let lastTrackingAt = 0;
        let trackingStartedAt = 0;
        let smoothedFit = null;
        let bodyTrackingFailed = false;
        let isSubmittingLiveAiFit = false;
        const liveFitAi = window.LiveFitAI || null;
        const liveFitConfig = liveFitAi?.config || {
            trackingIntervalMs: 90,
            poseModelComplexity: 0,
            drawArmsInFront: true,
            fallbackAfterMs: 1200
        };

        const resizeImageFile = (file, maxDimension = 800, quality = 0.82) => new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const image = new Image();
                image.onload = () => {
                    const ratio = Math.min(maxDimension / image.width, maxDimension / image.height, 1);
                    const width = Math.max(1, Math.round(image.width * ratio));
                    const height = Math.max(1, Math.round(image.height * ratio));
                    const resizeCanvas = document.createElement("canvas");
                    resizeCanvas.width = width;
                    resizeCanvas.height = height;
                    const context = resizeCanvas.getContext("2d");

                    if (!context) {
                        reject(new Error("Could not prepare image canvas."));
                        return;
                    }

                    context.drawImage(image, 0, 0, width, height);
                    resolve(resizeCanvas.toDataURL("image/jpeg", quality));
                };
                image.onerror = () => reject(new Error("Image could not be loaded."));
                image.src = reader.result;
            };
            reader.onerror = () => reject(new Error("Image could not be read."));
            reader.readAsDataURL(file);
        });

        const loadExternalScript = (src) => new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }

            const script = document.createElement("script");
            script.src = src;
            script.async = true;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Could not load ${src}`));
            document.head.appendChild(script);
        });

        const initializeBodyTracking = async () => {
            if (poseModel) {
                return;
            }

            if (!bodyTrackingPromise) {
                bodyTrackingPromise = loadExternalScript("https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js")
                .then(() => {
                    poseModel = new window.Pose({
                        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
                    });
                    poseModel.setOptions({
                        modelComplexity: liveFitConfig.poseModelComplexity,
                        smoothLandmarks: true,
                        enableSegmentation: true,
                        selfieMode: false,
                        minDetectionConfidence: 0.55,
                        minTrackingConfidence: 0.55
                    });
                    poseModel.onResults(results => {
                        latestPoseLandmarks = results.poseLandmarks || null;
                        latestSegmentationMask = results.segmentationMask || null;
                    });
                });
            }

            await bodyTrackingPromise;
        };

        const setPreviewState = (selector, emptySelector, src) => {
            form.querySelectorAll(selector).forEach(previewElement => {
                if (!src) {
                    previewElement.classList.add("d-none");
                    previewElement.removeAttribute("src");
                } else {
                    previewElement.src = src;
                    previewElement.classList.remove("d-none");
                }
            });

            form.querySelectorAll(emptySelector).forEach(emptyElement => {
                emptyElement.classList.toggle("d-none", Boolean(src));
            });
        };

        const showPreview = (file, previewSelector, emptySelector) => {
            if (!file) {
                setPreviewState(previewSelector, emptySelector, "");
                return;
            }

            setPreviewState(previewSelector, emptySelector, URL.createObjectURL(file));
        };

        const stopCamera = () => {
            trackingLoopActive = false;
            latestPoseLandmarks = null;
            latestSegmentationMask = null;
            smoothedFit = null;
            trackingStartedAt = 0;
            lastTrackingAt = 0;

            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }

            if (liveOverlayFrame) {
                cancelAnimationFrame(liveOverlayFrame);
                liveOverlayFrame = 0;
            }
        };

        const stopFileCamera = () => {
            if (fileCameraStream) {
                fileCameraStream.getTracks().forEach(track => track.stop());
                fileCameraStream = null;
            }

            if (fileCameraFeed) {
                fileCameraFeed.srcObject = null;
            }
        };

        const syncGarmentAreaFromCategory = () => {
            return syncCategoryCatalog(targetBody, clothingCategory, garmentArea);
        };

        const getGarmentArea = () => syncGarmentAreaFromCategory() || garmentArea?.value || "upper";
        const getGarmentCategory = () => (clothingCategory?.value || "").trim();
        const getGarmentView = () => {
            const value = (garmentView?.value || "front").trim().toLowerCase();
            return value === "back" ? "back" : "front";
        };

        const prepareOverlayCanvas = () => {
            const videoWidth = video.videoWidth || 0;
            const videoHeight = video.videoHeight || 0;
            const displayWidth = liveOverlayCanvas.clientWidth || video.clientWidth || videoWidth || 1;
            const displayHeight = liveOverlayCanvas.clientHeight || video.clientHeight || videoHeight || 1;

            if (liveOverlayCanvas.width !== displayWidth || liveOverlayCanvas.height !== displayHeight) {
                liveOverlayCanvas.width = displayWidth;
                liveOverlayCanvas.height = displayHeight;
            }

            return { displayWidth, displayHeight, videoWidth, videoHeight };
        };

        const videoCoverMetrics = (displayWidth, displayHeight) => {
            const videoWidth = video.videoWidth || displayWidth;
            const videoHeight = video.videoHeight || displayHeight;
            const scale = Math.max(displayWidth / videoWidth, displayHeight / videoHeight);
            const scaledWidth = videoWidth * scale;
            const scaledHeight = videoHeight * scale;
            return {
                scale,
                x: (displayWidth - scaledWidth) / 2,
                y: (displayHeight - scaledHeight) / 2,
                width: scaledWidth,
                height: scaledHeight,
                videoWidth,
                videoHeight
            };
        };

        const mapLandmarkToCanvas = (landmark, displayWidth, displayHeight) => {
            const cover = videoCoverMetrics(displayWidth, displayHeight);
            return {
                x: cover.x + (landmark.x * cover.videoWidth * cover.scale),
                y: cover.y + (landmark.y * cover.videoHeight * cover.scale),
                visibility: landmark.visibility ?? 1
            };
        };

        const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
        const midpoint = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
        const dot = (a, b) => (a.x * b.x) + (a.y * b.y);
        const addPoint = (point, vector, scale = 1) => ({
            x: point.x + (vector.x * scale),
            y: point.y + (vector.y * scale)
        });
        const normalizeVector = (vector, fallback = { x: 0, y: 1 }) => {
            const length = Math.hypot(vector.x, vector.y);
            if (!length || length < 0.0001) {
                return fallback;
            }

            return {
                x: vector.x / length,
                y: vector.y / length
            };
        };

        const validLandmarks = (landmarks) => {
            if (!landmarks) {
                return false;
            }

            const minVisibility = liveFitConfig.minLandmarkVisibility ?? 0.55;
            return [11, 12, 23, 24].every(index => (landmarks[index]?.visibility ?? 0) >= minVisibility);
        };

        const calculatePoseFit = (displayWidth, displayHeight) => {
            if (!validLandmarks(latestPoseLandmarks)) {
                return null;
            }

            const anatomicalLeftShoulder = mapLandmarkToCanvas(latestPoseLandmarks[11], displayWidth, displayHeight);
            const anatomicalRightShoulder = mapLandmarkToCanvas(latestPoseLandmarks[12], displayWidth, displayHeight);
            const anatomicalLeftHip = mapLandmarkToCanvas(latestPoseLandmarks[23], displayWidth, displayHeight);
            const anatomicalRightHip = mapLandmarkToCanvas(latestPoseLandmarks[24], displayWidth, displayHeight);
            const isMirroredPose = anatomicalLeftShoulder.x > anatomicalRightShoulder.x;
            const leftShoulder = isMirroredPose ? anatomicalRightShoulder : anatomicalLeftShoulder;
            const rightShoulder = isMirroredPose ? anatomicalLeftShoulder : anatomicalRightShoulder;
            const leftHip = isMirroredPose ? anatomicalRightHip : anatomicalLeftHip;
            const rightHip = isMirroredPose ? anatomicalLeftHip : anatomicalRightHip;
            const shoulderCenter = midpoint(leftShoulder, rightShoulder);
            const hipCenter = midpoint(leftHip, rightHip);
            const shoulderDistance = distance(leftShoulder, rightShoulder);
            const torsoDistance = distance(shoulderCenter, hipCenter);
            const minShoulderRatio = liveFitConfig.minShoulderRatio ?? 0.16;
            const maxShoulderRatio = liveFitConfig.maxShoulderRatio ?? 0.74;
            const minTorsoRatio = liveFitConfig.minTorsoRatio ?? 0.18;
            const maxTorsoRatio = liveFitConfig.maxTorsoRatio ?? 0.76;
            if (shoulderDistance < displayWidth * minShoulderRatio ||
                shoulderDistance > displayWidth * maxShoulderRatio ||
                torsoDistance < displayHeight * minTorsoRatio ||
                torsoDistance > displayHeight * maxTorsoRatio) {
                return null;
            }

            const shoulderVector = {
                x: rightShoulder.x - leftShoulder.x,
                y: rightShoulder.y - leftShoulder.y
            };
            const torsoVector = {
                x: hipCenter.x - shoulderCenter.x,
                y: hipCenter.y - shoulderCenter.y
            };
            const shoulderUnit = normalizeVector(shoulderVector, { x: 1, y: 0 });
            const torsoUnit = normalizeVector(torsoVector, { x: 0, y: 1 });
            let downUnit = normalizeVector({ x: -shoulderUnit.y, y: shoulderUnit.x }, torsoUnit);
            if (dot(downUnit, torsoUnit) < 0) {
                downUnit = { x: -downUnit.x, y: -downUnit.y };
            }
            downUnit = normalizeVector({
                x: (downUnit.x * 0.74) + (torsoUnit.x * 0.26),
                y: (downUnit.y * 0.74) + (torsoUnit.y * 0.26)
            }, downUnit);

            const hipDistance = Math.max(distance(leftHip, rightHip), shoulderDistance * 0.70);
            const mapOrEstimate = (index, estimate) => {
                const landmark = latestPoseLandmarks[index];
                return (landmark?.visibility ?? 0) > 0.30
                    ? mapLandmarkToCanvas(landmark, displayWidth, displayHeight)
                    : estimate;
            };
            const mirroredIndex = (normalIndex, mirroredIndex) => isMirroredPose ? mirroredIndex : normalIndex;
            const leftElbow = mapOrEstimate(mirroredIndex(13, 14), {
                x: leftShoulder.x - (shoulderDistance * 0.22),
                y: leftShoulder.y + (torsoDistance * 0.48),
                visibility: 0.30
            });
            const rightElbow = mapOrEstimate(mirroredIndex(14, 13), {
                x: rightShoulder.x + (shoulderDistance * 0.22),
                y: rightShoulder.y + (torsoDistance * 0.48),
                visibility: 0.30
            });
            const leftWrist = mapOrEstimate(mirroredIndex(15, 16), {
                x: leftElbow.x - (shoulderDistance * 0.16),
                y: leftElbow.y + (torsoDistance * 0.54),
                visibility: 0.25
            });
            const rightWrist = mapOrEstimate(mirroredIndex(16, 15), {
                x: rightElbow.x + (shoulderDistance * 0.16),
                y: rightElbow.y + (torsoDistance * 0.54),
                visibility: 0.25
            });
            const leftKneeEstimate = {
                x: leftHip.x + ((leftHip.x - leftShoulder.x) * 0.26),
                y: leftHip.y + (torsoDistance * 0.88),
                visibility: 0.30
            };
            const rightKneeEstimate = {
                x: rightHip.x + ((rightHip.x - rightShoulder.x) * 0.26),
                y: rightHip.y + (torsoDistance * 0.88),
                visibility: 0.30
            };
            const leftKnee = mapOrEstimate(mirroredIndex(25, 26), leftKneeEstimate);
            const rightKnee = mapOrEstimate(mirroredIndex(26, 25), rightKneeEstimate);
            const kneeCenter = midpoint(leftKnee, rightKnee);
            const leftAnkle = mapOrEstimate(mirroredIndex(27, 28), {
                x: leftKnee.x + ((leftKnee.x - leftHip.x) * 0.24),
                y: leftKnee.y + (torsoDistance * 0.92),
                visibility: 0.25
            });
            const rightAnkle = mapOrEstimate(mirroredIndex(28, 27), {
                x: rightKnee.x + ((rightKnee.x - rightHip.x) * 0.24),
                y: rightKnee.y + (torsoDistance * 0.92),
                visibility: 0.25
            });
            const ankleCenter = midpoint(leftAnkle, rightAnkle);
            const rawAngle = Math.atan2(shoulderUnit.y, shoulderUnit.x);
            const angle = liveFitAi?.clampAngle?.(rawAngle) ?? rawAngle;
            const category = getGarmentCategory();
            const area = getGarmentArea();
            const fitProfile = liveFitAi?.getProfile?.(category, area) ||
                liveFitAi?.getFactors?.(area) || {
                    area,
                    anchor: area,
                    widthFactor: area === "overall" ? 1.34 : area === "lower" ? 1.18 : 1.28,
                    heightFactor: area === "overall" ? 1.03 : area === "lower" ? 1.06 : 1.08,
                    centerFactor: area === "overall" ? 0.53 : area === "lower" ? 0.52 : 0.48
                };
            const anchor = fitProfile.anchor || fitProfile.area || area;
            const anchorEnd = fitProfile.length === "short" ? kneeCenter : ankleCenter;
            const startCenter = anchor === "lower" ? hipCenter : shoulderCenter;
            const endCenter = anchor === "upper" ? hipCenter : anchorEnd;
            const anchorVector = {
                x: endCenter.x - startCenter.x,
                y: endCenter.y - startCenter.y
            };
            const anchorDistance = Math.max(80, distance(startCenter, endCenter));
            const widthBasis = anchor === "lower"
                ? Math.max(hipDistance, shoulderDistance * 0.72)
                : anchor === "overall"
                    ? Math.max(shoulderDistance * 1.02, hipDistance * 1.16)
                    : Math.max(shoulderDistance, hipDistance * 0.96);
            const rawGarmentHeight = Math.max(fitProfile.minHeight || 110, anchorDistance * fitProfile.heightFactor);
            const garmentHeight = anchor === "upper" && fitProfile.maxHeightFromShoulders
                ? Math.min(rawGarmentHeight, shoulderDistance * fitProfile.maxHeightFromShoulders)
                : rawGarmentHeight;
            const neckLift = shoulderDistance * (fitProfile.neckLiftFactor ?? 0.08);
            const neckAnchor = anchor === "lower"
                ? hipCenter
                : {
                    x: shoulderCenter.x - (downUnit.x * neckLift),
                    y: shoulderCenter.y - (downUnit.y * neckLift)
                };
            const collarToCenter = (0.5 - (fitProfile.collarY ?? 0.17)) * garmentHeight;
            const anchoredCenter = anchor === "lower"
                ? {
                    x: startCenter.x + (anchorVector.x * fitProfile.centerFactor),
                    y: startCenter.y + (anchorVector.y * fitProfile.centerFactor)
                }
                : {
                    x: neckAnchor.x + (downUnit.x * collarToCenter),
                    y: neckAnchor.y + (downUnit.y * collarToCenter)
                };

            const rawFit = {
                area: fitProfile.area || area,
                category: liveFitAi?.normalizeCategory?.(category) || category.toLowerCase(),
                viewDirection: getGarmentView(),
                profile: fitProfile,
                leftShoulder,
                rightShoulder,
                leftHip,
                rightHip,
                leftElbow,
                rightElbow,
                leftWrist,
                rightWrist,
                leftKnee,
                rightKnee,
                leftAnkle,
                rightAnkle,
                shoulderCenter,
                hipCenter,
                kneeCenter,
                ankleCenter,
                neck: neckAnchor,
                center: anchoredCenter,
                shoulderUnit,
                downUnit,
                angle,
                shoulderDistance,
                hipDistance,
                torsoDistance,
                anchorDistance,
                width: Math.max(fitProfile.minWidth || 80, widthBasis * fitProfile.widthFactor),
                height: garmentHeight
            };
            const targetFit = liveFitAi?.normalizeFit?.(rawFit, displayWidth, displayHeight) || rawFit;

            if (!smoothedFit ||
                smoothedFit.category !== targetFit.category ||
                smoothedFit.area !== targetFit.area ||
                smoothedFit.viewDirection !== targetFit.viewDirection) {
                smoothedFit = targetFit;
                return targetFit;
            }

            const alpha = 0.22;
            const smoothPoint = (previous, next) => ({
                x: (previous?.x ?? next.x) + ((next.x - (previous?.x ?? next.x)) * alpha),
                y: (previous?.y ?? next.y) + ((next.y - (previous?.y ?? next.y)) * alpha),
                visibility: next.visibility
            });
            const smoothUnit = (previous, next) => normalizeVector({
                x: (previous?.x ?? next.x) + ((next.x - (previous?.x ?? next.x)) * alpha),
                y: (previous?.y ?? next.y) + ((next.y - (previous?.y ?? next.y)) * alpha)
            }, next);

            smoothedFit = {
                ...targetFit,
                leftShoulder: smoothPoint(smoothedFit.leftShoulder, targetFit.leftShoulder),
                rightShoulder: smoothPoint(smoothedFit.rightShoulder, targetFit.rightShoulder),
                leftHip: smoothPoint(smoothedFit.leftHip, targetFit.leftHip),
                rightHip: smoothPoint(smoothedFit.rightHip, targetFit.rightHip),
                leftElbow: smoothPoint(smoothedFit.leftElbow, targetFit.leftElbow),
                rightElbow: smoothPoint(smoothedFit.rightElbow, targetFit.rightElbow),
                leftWrist: smoothPoint(smoothedFit.leftWrist, targetFit.leftWrist),
                rightWrist: smoothPoint(smoothedFit.rightWrist, targetFit.rightWrist),
                leftKnee: smoothPoint(smoothedFit.leftKnee, targetFit.leftKnee),
                rightKnee: smoothPoint(smoothedFit.rightKnee, targetFit.rightKnee),
                leftAnkle: smoothPoint(smoothedFit.leftAnkle, targetFit.leftAnkle),
                rightAnkle: smoothPoint(smoothedFit.rightAnkle, targetFit.rightAnkle),
                shoulderCenter: smoothPoint(smoothedFit.shoulderCenter, targetFit.shoulderCenter),
                hipCenter: smoothPoint(smoothedFit.hipCenter, targetFit.hipCenter),
                kneeCenter: smoothPoint(smoothedFit.kneeCenter, targetFit.kneeCenter),
                ankleCenter: smoothPoint(smoothedFit.ankleCenter, targetFit.ankleCenter),
                neck: smoothPoint(smoothedFit.neck, targetFit.neck),
                center: smoothPoint(smoothedFit.center, targetFit.center),
                shoulderUnit: smoothUnit(smoothedFit.shoulderUnit, targetFit.shoulderUnit),
                downUnit: smoothUnit(smoothedFit.downUnit, targetFit.downUnit),
                angle: smoothedFit.angle + ((targetFit.angle - smoothedFit.angle) * alpha),
                shoulderDistance: smoothedFit.shoulderDistance + ((targetFit.shoulderDistance - smoothedFit.shoulderDistance) * alpha),
                hipDistance: smoothedFit.hipDistance + ((targetFit.hipDistance - smoothedFit.hipDistance) * alpha),
                torsoDistance: smoothedFit.torsoDistance + ((targetFit.torsoDistance - smoothedFit.torsoDistance) * alpha),
                anchorDistance: (smoothedFit.anchorDistance || targetFit.anchorDistance) + ((targetFit.anchorDistance - (smoothedFit.anchorDistance || targetFit.anchorDistance)) * alpha),
                width: smoothedFit.width + ((targetFit.width - smoothedFit.width) * alpha),
                height: smoothedFit.height + ((targetFit.height - smoothedFit.height) * alpha)
            };

            return liveFitAi?.normalizeFit?.(smoothedFit, displayWidth, displayHeight) || smoothedFit;
        };

        const drawTransformedGarment = (context, fit, garmentImage, alpha = 0.94) => {
            const garmentRatio = garmentImage.width / Math.max(1, garmentImage.height);
            const targetRatio = fit.width / Math.max(1, fit.height);
            let drawWidth = fit.width;
            let drawHeight = fit.height;

            if (fit.area === "upper" || fit.area === "overall") {
                drawWidth = fit.width;
                drawHeight = drawWidth / Math.max(0.1, garmentRatio);
                const maxHeight = fit.height * 1.16;
                if (drawHeight > maxHeight) {
                    drawHeight = maxHeight;
                    drawWidth = drawHeight * garmentRatio;
                }
            } else if (garmentRatio > targetRatio) {
                drawHeight = drawWidth / garmentRatio;
            } else {
                drawWidth = drawHeight * garmentRatio;
            }

            const profile = fit.profile || {};
            const isBackView = fit.viewDirection === "back";
            const downUnit = normalizeVector(fit.downUnit || {
                x: -Math.sin(fit.angle || 0),
                y: Math.cos(fit.angle || 0)
            });
            const shoulderUnit = normalizeVector(fit.shoulderUnit || {
                x: Math.cos(fit.angle || 0),
                y: Math.sin(fit.angle || 0)
            }, { x: 1, y: 0 });
            const collarY = profile.collarY ?? 0.17;
            const drawCenter = (fit.area === "upper" || fit.area === "overall") && fit.neck
                ? {
                    x: fit.neck.x + (downUnit.x * drawHeight * (0.5 - collarY)),
                    y: fit.neck.y + (downUnit.y * drawHeight * (0.5 - collarY))
                }
                : fit.area === "lower" && fit.kneeCenter && profile.kneeY
                    ? {
                        x: fit.kneeCenter.x - (downUnit.x * drawHeight * (profile.kneeY - 0.5)),
                        y: fit.kneeCenter.y - (downUnit.y * drawHeight * (profile.kneeY - 0.5))
                    }
                : fit.center;

            if (profile.frontPanelOnly && fit.area === "upper") {
                const imageWidth = garmentImage.naturalWidth || garmentImage.width;
                const imageHeight = garmentImage.naturalHeight || garmentImage.height;
                const crop = (isBackView && profile.backCrop) ||
                    profile.frontCrop ||
                    { x: 0.18, y: 0.08, width: 0.64, height: 0.84 };
                const sx = imageWidth * crop.x;
                const sy = imageHeight * crop.y;
                const sw = imageWidth * crop.width;
                const sh = imageHeight * crop.height;
                const panelHeight = Math.min(
                    fit.torsoDistance * (profile.torsoHeightFactor ?? 1.24),
                    fit.shoulderDistance * (profile.maxHeightFromShoulders ?? 1.82));
                const topWidth = fit.shoulderDistance * (profile.torsoTopWidthFactor ?? 1.10);
                const bottomWidth = Math.max(
                    fit.hipDistance * (profile.torsoBottomWidthFactor ?? 0.96),
                    topWidth * 0.78);
                const collarToCenter = panelHeight * (0.5 - collarY);
                const panelCenter = fit.neck
                    ? {
                        x: fit.neck.x + (downUnit.x * collarToCenter),
                        y: fit.neck.y + (downUnit.y * collarToCenter)
                    }
                    : drawCenter;

                context.save();
                context.transform(shoulderUnit.x, shoulderUnit.y, downUnit.x, downUnit.y, panelCenter.x, panelCenter.y);
                context.globalAlpha = fit.profile?.opacity || alpha;
                context.shadowColor = "rgba(0, 0, 0, 0.20)";
                context.shadowBlur = 9;
                context.shadowOffsetY = 5;
                context.beginPath();
                context.moveTo(-topWidth / 2, -panelHeight / 2);
                context.lineTo(topWidth / 2, -panelHeight / 2);
                context.lineTo(bottomWidth / 2, panelHeight / 2);
                context.lineTo(-bottomWidth / 2, panelHeight / 2);
                context.closePath();
                context.clip();
                context.drawImage(garmentImage, sx, sy, sw, sh, -bottomWidth / 2, -panelHeight / 2, bottomWidth, panelHeight);
                context.shadowColor = "transparent";
                context.globalCompositeOperation = "multiply";
                const torsoShade = context.createLinearGradient(-bottomWidth / 2, 0, bottomWidth / 2, 0);
                torsoShade.addColorStop(0, "rgba(0, 0, 0, 0.24)");
                torsoShade.addColorStop(0.24, "rgba(0, 0, 0, 0.06)");
                torsoShade.addColorStop(0.50, "rgba(255, 255, 255, 0.02)");
                torsoShade.addColorStop(0.76, "rgba(0, 0, 0, 0.06)");
                torsoShade.addColorStop(1, "rgba(0, 0, 0, 0.22)");
                context.fillStyle = torsoShade;
                context.fillRect(-bottomWidth / 2, -panelHeight / 2, bottomWidth, panelHeight);
                context.globalCompositeOperation = "screen";
                const fabricHighlight = context.createRadialGradient(0, -panelHeight * 0.06, 4, 0, -panelHeight * 0.06, topWidth * 0.52);
                fabricHighlight.addColorStop(0, "rgba(255, 255, 255, 0.10)");
                fabricHighlight.addColorStop(1, "rgba(255, 255, 255, 0)");
                context.fillStyle = fabricHighlight;
                context.fillRect(-bottomWidth / 2, -panelHeight / 2, bottomWidth, panelHeight);
                context.restore();

                if (fit.profile?.drawPoseSleeves !== false) {
                    drawPoseSleeves(context, fit, garmentImage, bottomWidth, panelHeight, alpha);
                }

                return;
            }

            context.save();
            context.transform(shoulderUnit.x, shoulderUnit.y, downUnit.x, downUnit.y, drawCenter.x, drawCenter.y);
            context.globalAlpha = fit.profile?.opacity || alpha;
            context.shadowColor = "rgba(0, 0, 0, 0.18)";
            context.shadowBlur = fit.area === "overall" ? 12 : 10;
            context.shadowOffsetY = fit.area === "lower" ? 5 : 6;
            context.drawImage(garmentImage, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
            if (fit.area === "upper" || fit.area === "overall") {
                context.shadowColor = "transparent";
                context.globalCompositeOperation = "multiply";
                const sideShade = context.createLinearGradient(-drawWidth / 2, 0, drawWidth / 2, 0);
                sideShade.addColorStop(0, "rgba(0, 0, 0, 0.20)");
                sideShade.addColorStop(0.22, "rgba(0, 0, 0, 0.05)");
                sideShade.addColorStop(0.50, "rgba(255, 255, 255, 0.03)");
                sideShade.addColorStop(0.78, "rgba(0, 0, 0, 0.05)");
                sideShade.addColorStop(1, "rgba(0, 0, 0, 0.18)");
                context.fillStyle = sideShade;
                context.fillRect(-drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
                context.globalCompositeOperation = "screen";
                const chestHighlight = context.createRadialGradient(0, -drawHeight * 0.08, 4, 0, -drawHeight * 0.08, drawWidth * 0.46);
                chestHighlight.addColorStop(0, "rgba(255, 255, 255, 0.10)");
                chestHighlight.addColorStop(1, "rgba(255, 255, 255, 0)");
                context.fillStyle = chestHighlight;
                context.fillRect(-drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
            }
            context.restore();

            if (fit.area === "upper" && fit.profile?.drawPoseSleeves !== false) {
                drawPoseSleeves(context, fit, garmentImage, drawWidth, drawHeight, alpha);
            }
        };

        const drawImageInQuad = (context, image, sourceRect, p0, p1, p2, p3, alpha = 0.94) => {
            const sw = Math.max(1, sourceRect.sw);
            const sh = Math.max(1, sourceRect.sh);

            context.save();
            context.beginPath();
            context.moveTo(p0.x, p0.y);
            context.lineTo(p1.x, p1.y);
            context.lineTo(p2.x, p2.y);
            context.lineTo(p3.x, p3.y);
            context.closePath();
            context.clip();
            context.globalAlpha = alpha;
            context.transform(
                (p1.x - p0.x) / sw,
                (p1.y - p0.y) / sw,
                (p3.x - p0.x) / sh,
                (p3.y - p0.y) / sh,
                p0.x,
                p0.y);
            context.drawImage(image, sourceRect.sx, sourceRect.sy, sw, sh, 0, 0, sw, sh);
            context.globalCompositeOperation = "multiply";
            const sleeveShade = context.createLinearGradient(0, 0, sw, sh);
            sleeveShade.addColorStop(0, "rgba(255, 255, 255, 0.03)");
            sleeveShade.addColorStop(0.68, "rgba(0, 0, 0, 0.08)");
            sleeveShade.addColorStop(1, "rgba(0, 0, 0, 0.18)");
            context.fillStyle = sleeveShade;
            context.fillRect(0, 0, sw, sh);
            context.restore();
        };

        const drawPoseSleeves = (context, fit, garmentImage, drawWidth, drawHeight, alpha = 0.94) => {
            const profile = fit.profile || {};
            const shoulderUnit = normalizeVector(fit.shoulderUnit || { x: 1, y: 0 }, { x: 1, y: 0 });
            const downUnit = normalizeVector(fit.downUnit || { x: 0, y: 1 }, { x: 0, y: 1 });
            const imageWidth = garmentImage.naturalWidth || garmentImage.width;
            const imageHeight = garmentImage.naturalHeight || garmentImage.height;
            const sleeveLengthFactor = profile.sleeveLengthFactor ?? 0.82;
            const sleeveWidthFactor = profile.sleeveWidthFactor ?? 0.18;
            const shoulderOffset = profile.sleeveShoulderOffset ?? 0.02;
            const sleeveMode = profile.sleeveMode || "short";

            const drawOneSleeve = (side) => {
                const isLeft = side === "left";
                const shoulder = isLeft ? fit.leftShoulder : fit.rightShoulder;
                const elbow = isLeft ? fit.leftElbow : fit.rightElbow;
                const wrist = isLeft ? fit.leftWrist : fit.rightWrist;
                if (!shoulder || !elbow || (elbow.visibility ?? 1) < 0.22) {
                    return;
                }

                let outward = isLeft
                    ? { x: -shoulderUnit.x, y: -shoulderUnit.y }
                    : { x: shoulderUnit.x, y: shoulderUnit.y };
                outward = normalizeVector(outward, isLeft ? { x: -1, y: 0 } : { x: 1, y: 0 });

                const armFallback = normalizeVector({
                    x: (downUnit.x * 0.72) + (outward.x * 0.28),
                    y: (downUnit.y * 0.72) + (outward.y * 0.28)
                }, downUnit);

                const startHalf = fit.shoulderDistance * sleeveWidthFactor;
                const startCenter = addPoint(
                    addPoint(shoulder, outward, fit.shoulderDistance * shoulderOffset),
                    downUnit,
                    fit.shoulderDistance * 0.035);

                const drawSleeveSegment = (from, to, fromHalf, toHalf, sourceY, sourceHeight, minLength) => {
                    const segmentVector = {
                        x: to.x - from.x,
                        y: to.y - from.y
                    };
                    const segmentUnit = normalizeVector(segmentVector, armFallback);
                    let segmentNormal = normalizeVector({ x: -segmentUnit.y, y: segmentUnit.x }, outward);
                    if (dot(segmentNormal, outward) < 0) {
                        segmentNormal = { x: -segmentNormal.x, y: -segmentNormal.y };
                    }

                    const segmentDistance = Math.max(distance(from, to), minLength);
                    const segmentEnd = addPoint(from, segmentUnit, segmentDistance);
                    const p0 = addPoint(from, segmentNormal, fromHalf);
                    const p1 = addPoint(from, segmentNormal, -fromHalf);
                    const p2 = addPoint(segmentEnd, segmentNormal, -toHalf);
                    const p3 = addPoint(segmentEnd, segmentNormal, toHalf);
                    const sourceRect = {
                        sx: imageWidth * (isLeft ? 0.00 : 0.64),
                        sy: imageHeight * sourceY,
                        sw: imageWidth * 0.36,
                        sh: imageHeight * sourceHeight
                    };

                    drawImageInQuad(context, garmentImage, sourceRect, p0, p1, p2, p3, Math.min(alpha, 0.95));
                };

                if (sleeveMode === "long" && wrist && (wrist.visibility ?? 1) >= 0.18) {
                    drawSleeveSegment(
                        startCenter,
                        elbow,
                        startHalf,
                        startHalf * 0.76,
                        0.10,
                        0.32,
                        fit.shoulderDistance * 0.28);
                    drawSleeveSegment(
                        elbow,
                        wrist,
                        startHalf * 0.74,
                        startHalf * 0.50,
                        0.38,
                        0.34,
                        fit.shoulderDistance * 0.26);
                    return;
                }

                const shoulderToElbow = distance(startCenter, elbow);
                const shortEnd = addPoint(
                    startCenter,
                    normalizeVector({ x: elbow.x - startCenter.x, y: elbow.y - startCenter.y }, armFallback),
                    Math.min(
                        Math.max(shoulderToElbow * sleeveLengthFactor, fit.shoulderDistance * 0.24),
                        fit.shoulderDistance * 0.78));
                drawSleeveSegment(
                    startCenter,
                    shortEnd,
                    startHalf,
                    startHalf * 0.64,
                    0.12,
                    0.42,
                    fit.shoulderDistance * 0.22);
            };

            drawOneSleeve("left");
            drawOneSleeve("right");
        };

        const drawClippedGarment = (context, fit, displayWidth, displayHeight) => {
            if (!latestSegmentationMask) {
                drawTransformedGarment(context, fit, liveGarmentImage);
                return;
            }

            const garmentLayer = document.createElement("canvas");
            garmentLayer.width = displayWidth;
            garmentLayer.height = displayHeight;
            const garmentContext = garmentLayer.getContext("2d");
            drawTransformedGarment(garmentContext, fit, liveGarmentImage);
            garmentContext.globalCompositeOperation = "destination-in";
            garmentContext.drawImage(latestSegmentationMask, 0, 0, displayWidth, displayHeight);
            garmentContext.globalCompositeOperation = "source-over";

            context.drawImage(garmentLayer, 0, 0);
        };

        const drawVideoCover = (context, displayWidth, displayHeight) => {
            const cover = videoCoverMetrics(displayWidth, displayHeight);
            context.drawImage(video, cover.x, cover.y, cover.width, cover.height);
        };

        const drawBodyPartsInFront = (context, displayWidth, displayHeight, fit) => {
            if (!latestPoseLandmarks || !liveFitConfig.drawArmsInFront) {
                return;
            }

            const armPairs = [
                [13, 15],
                [14, 16]
            ];

            const maskCanvas = document.createElement("canvas");
            maskCanvas.width = displayWidth;
            maskCanvas.height = displayHeight;
            const maskContext = maskCanvas.getContext("2d");
            maskContext.lineCap = "round";
            maskContext.lineJoin = "round";
            maskContext.lineWidth = Math.max(24, fit.shoulderDistance * 0.18);
            maskContext.strokeStyle = "#fff";
            maskContext.beginPath();

            armPairs.forEach(pair => {
                const points = pair.map(index => mapLandmarkToCanvas(latestPoseLandmarks[index], displayWidth, displayHeight));
                if (points.every(point => point.visibility > 0.35)) {
                    maskContext.moveTo(points[0].x, points[0].y);
                    maskContext.lineTo(points[1].x, points[1].y);
                }
            });

            maskContext.stroke();

            const armLayer = document.createElement("canvas");
            armLayer.width = displayWidth;
            armLayer.height = displayHeight;
            const armContext = armLayer.getContext("2d");
            drawVideoCover(armContext, displayWidth, displayHeight);
            armContext.globalCompositeOperation = "destination-in";
            armContext.drawImage(maskCanvas, 0, 0);
            armContext.globalCompositeOperation = "source-over";

            context.drawImage(armLayer, 0, 0);
        };

        const drawPoseDebug = (context, fit) => {
            if (!poseDebugToggle?.checked) {
                return;
            }

            const points = [
                ["LS", fit.leftShoulder],
                ["RS", fit.rightShoulder],
                ["LH", fit.leftHip],
                ["RH", fit.rightHip],
                ["NECK", fit.neck]
            ];

            context.save();
            context.font = "700 12px Manrope, Segoe UI, sans-serif";
            points.forEach(([label, point]) => {
                context.beginPath();
                context.arc(point.x, point.y, 7, 0, Math.PI * 2);
                context.fillStyle = label === "NECK" ? "#4ade80" : "#f97316";
                context.fill();
                context.fillStyle = "#fff";
                context.fillText(label, point.x + 10, point.y - 8);
            });

            context.strokeStyle = "rgba(74, 222, 128, 0.95)";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(fit.leftShoulder.x, fit.leftShoulder.y);
            context.lineTo(fit.rightShoulder.x, fit.rightShoulder.y);
            context.moveTo(fit.leftHip.x, fit.leftHip.y);
            context.lineTo(fit.rightHip.x, fit.rightHip.y);
            context.stroke();
            context.restore();
        };

        const calculateFallbackFit = (displayWidth, displayHeight) => {
            const shoulderDistance = displayWidth * 0.27;
            const torsoDistance = displayHeight * 0.28;
            const hipDistance = shoulderDistance * 0.84;
            const category = getGarmentCategory();
            const area = getGarmentArea();
            const fitProfile = liveFitAi?.getProfile?.(category, area) ||
                liveFitAi?.getFactors?.(area) ||
                { area, anchor: area, widthFactor: 1.28, heightFactor: 1.08, centerFactor: 0.48 };
            const anchor = fitProfile.anchor || fitProfile.area || area;
            const fallbackFit = {
                area: fitProfile.area || area,
                category: liveFitAi?.normalizeCategory?.(category) || category.toLowerCase(),
                viewDirection: getGarmentView(),
                profile: fitProfile,
                leftShoulder: { x: (displayWidth / 2) - (shoulderDistance / 2), y: displayHeight * 0.31 },
                rightShoulder: { x: (displayWidth / 2) + (shoulderDistance / 2), y: displayHeight * 0.31 },
                leftHip: { x: (displayWidth / 2) - (shoulderDistance * 0.42), y: displayHeight * 0.56 },
                rightHip: { x: (displayWidth / 2) + (shoulderDistance * 0.42), y: displayHeight * 0.56 },
                leftKnee: { x: (displayWidth / 2) - (shoulderDistance * 0.34), y: displayHeight * 0.73 },
                rightKnee: { x: (displayWidth / 2) + (shoulderDistance * 0.34), y: displayHeight * 0.73 },
                leftAnkle: { x: (displayWidth / 2) - (shoulderDistance * 0.30), y: displayHeight * 0.88 },
                rightAnkle: { x: (displayWidth / 2) + (shoulderDistance * 0.30), y: displayHeight * 0.88 },
                shoulderCenter: { x: displayWidth / 2, y: displayHeight * 0.31 },
                hipCenter: { x: displayWidth / 2, y: displayHeight * 0.56 },
                kneeCenter: { x: displayWidth / 2, y: displayHeight * 0.73 },
                ankleCenter: { x: displayWidth / 2, y: displayHeight * 0.88 },
                neck: {
                    x: displayWidth / 2,
                    y: (displayHeight * 0.31) - (shoulderDistance * (fitProfile.neckLiftFactor ?? 0.08))
                },
                shoulderUnit: { x: 1, y: 0 },
                downUnit: { x: 0, y: 1 },
                angle: 0,
                shoulderDistance,
                hipDistance,
                torsoDistance,
                anchorDistance: torsoDistance
            };
            const startCenter = anchor === "lower" ? fallbackFit.hipCenter : fallbackFit.shoulderCenter;
            const endCenter = anchor === "upper"
                ? fallbackFit.hipCenter
                : fitProfile.length === "short"
                    ? fallbackFit.kneeCenter
                    : fallbackFit.ankleCenter;
            const anchorDistance = distance(startCenter, endCenter);
            const widthBasis = anchor === "lower"
                ? Math.max(hipDistance, shoulderDistance * 0.72)
                : anchor === "overall"
                    ? Math.max(shoulderDistance * 1.02, hipDistance * 1.16)
                    : Math.max(shoulderDistance, hipDistance * 0.96);
            const rawHeight = Math.max(fitProfile.minHeight || 110, anchorDistance * fitProfile.heightFactor);
            fallbackFit.anchorDistance = anchorDistance;
            fallbackFit.width = Math.max(fitProfile.minWidth || 80, widthBasis * fitProfile.widthFactor);
            fallbackFit.height = anchor === "upper" && fitProfile.maxHeightFromShoulders
                ? Math.min(rawHeight, shoulderDistance * fitProfile.maxHeightFromShoulders)
                : rawHeight;
            fallbackFit.center = anchor === "lower"
                ? {
                    x: startCenter.x + ((endCenter.x - startCenter.x) * fitProfile.centerFactor),
                    y: startCenter.y + ((endCenter.y - startCenter.y) * fitProfile.centerFactor)
                }
                : {
                    x: fallbackFit.neck.x,
                    y: fallbackFit.neck.y + (fallbackFit.height * (0.5 - (fitProfile.collarY ?? 0.17)))
                };

            return liveFitAi?.normalizeFit?.(fallbackFit, displayWidth, displayHeight) || fallbackFit;
        };

        const updateDebugOutput = (fit) => {
            if (!fittingDebugOutput) {
                return;
            }

            if (!fit) {
                fittingDebugOutput.textContent = "Body not detected";
                return;
            }

            const label = `${fit.viewDirection === "back" ? "back" : "front"} ${fit.category || fit.area || "garment"}`;
            const text = `${label} ${Math.round(fit.width)} x ${Math.round(fit.height)} | shoulders ${Math.round(fit.shoulderDistance)} | torso ${Math.round(fit.torsoDistance)}`;
            fittingDebugOutput.textContent = text;
            console.debug("[try-on fit]", text, fit);
        };

        const processBodyTrackingLoop = async () => {
            const now = performance.now();
            if (!trackingLoopActive ||
                !stream ||
                !video.videoWidth ||
                !poseModel ||
                isTrackingFrame ||
                now - lastTrackingAt < liveFitConfig.trackingIntervalMs) {
                return;
            }

            try {
                isTrackingFrame = true;
                lastTrackingAt = now;
                await poseModel.send({ image: video });
            } catch (error) {
                console.warn("Body tracking failed", error);
                cameraStatus.textContent = "Body tracking unavailable";
            } finally {
                isTrackingFrame = false;
            }
        };

        const drawLiveGarment = () => {
            if (!liveOverlayCanvas || !video || !stream) {
                liveOverlayFrame = requestAnimationFrame(drawLiveGarment);
                return;
            }

            const { displayWidth, displayHeight, videoWidth, videoHeight } = prepareOverlayCanvas();
            const context = liveOverlayCanvas.getContext("2d");
            context.clearRect(0, 0, displayWidth, displayHeight);

            if (!videoWidth || !videoHeight || !liveGarmentImage) {
                if (liveGarmentUrl) {
                    cameraStatus.textContent = "Garment ready - press AI Fit";
                }
                liveOverlayFrame = requestAnimationFrame(drawLiveGarment);
                return;
            }

            processBodyTrackingLoop();
            const poseFit = calculatePoseFit(displayWidth, displayHeight);
            const shouldUseFallback = trackingStartedAt &&
                performance.now() - trackingStartedAt > liveFitConfig.fallbackAfterMs;
            const fit = poseFit || smoothedFit ||
                (shouldUseFallback ? calculateFallbackFit(displayWidth, displayHeight) : null);
            updateDebugOutput(fit);

            if (fit) {
                drawClippedGarment(context, fit, displayWidth, displayHeight);
                drawBodyPartsInFront(context, displayWidth, displayHeight, fit);
                drawPoseDebug(context, fit);
                cameraStatus.textContent = poseFit
                    ? "Body tracked - live fit running"
                    : bodyTrackingFailed
                        ? "Basic live fit running"
                        : "Tracking fallback - keep shoulders visible";
            } else {
                cameraStatus.textContent = "Step back until shoulders and hips are visible";
            }

            liveOverlayFrame = requestAnimationFrame(drawLiveGarment);
        };

        const startLiveOverlay = async () => {
            try {
                await initializeBodyTracking();
                trackingLoopActive = true;
                trackingStartedAt = performance.now();
                lastTrackingAt = 0;
                bodyTrackingFailed = false;
            } catch (error) {
                console.warn("MediaPipe could not start", error);
                bodyTrackingFailed = true;
                trackingStartedAt = performance.now();
                cameraStatus.textContent = "MediaPipe loading failed - using basic preview";
            }

            if (!liveOverlayFrame) {
                liveOverlayFrame = requestAnimationFrame(drawLiveGarment);
            }
        };

        const loadLiveGarment = (src, allowCanvasDraw = true) => {
            liveGarmentUrl = src || "";
            liveGarmentImage = null;
            smoothedFit = null;

            if (!src) {
                liveGarmentOverlay?.classList.add("d-none");
                cameraStatus.textContent = stream ? "Live camera running" : "Live camera ready";
                return;
            }

            const image = new Image();
            image.onload = () => {
                liveGarmentImage = liveFitAi?.prepareGarmentImage?.(image) || image;
                liveGarmentOverlay?.classList.add("d-none");
                cameraStatus.textContent = "Live try-on preview running";
                startLiveOverlay();
            };

            image.onerror = () => {
                if (allowCanvasDraw) {
                    liveGarmentOverlay.src = liveGarmentUrl;
                    liveGarmentOverlay.classList.remove("d-none");
                }
                cameraStatus.textContent = allowCanvasDraw
                    ? "Preview loaded with basic overlay"
                    : "Garment link ready - press AI Fit";
            };

            image.src = src;
        };

        const setMode = (mode) => {
            const useCamera = mode === "camera";
            const useUpload = mode === "upload";
            const useUrl = mode === "url";
            cameraSection.classList.toggle("d-none", !useCamera);
            posePreviewShell?.classList.toggle("d-none", useCamera);
            poseUploadSection?.classList.toggle("d-none", !useUpload);
            poseUrlSection?.classList.toggle("d-none", !useUrl);

            sourceButtons.forEach(button => {
                button.classList.toggle("is-active", button.dataset.sourceTrigger === mode);
            });

            if (!useCamera) {
                stopCamera();
            }
        };

        sourceButtons.forEach(button => {
            button.addEventListener("click", () => setMode(button.dataset.sourceTrigger));
        });

        poseUploadTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            poseLinkPanel?.classList.add("d-none");
            poseUpload?.click();
        }));

        clothingUploadTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            clothingLinkPanel?.classList.add("d-none");
            clothingUpload?.click();
        }));

        liveClothingUploadTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            liveClothingLinkPanel?.classList.add("d-none");
            clothingUpload?.click();
        }));

        poseLinkTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            poseLinkPanel?.classList.toggle("d-none");
            if (!poseLinkPanel?.classList.contains("d-none")) {
                poseUrl?.focus();
            }
        }));

        clothingLinkTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            clothingLinkPanel?.classList.toggle("d-none");
            if (!clothingLinkPanel?.classList.contains("d-none")) {
                clothingUrl?.focus();
            }
        }));

        liveClothingLinkTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            liveClothingLinkPanel?.classList.toggle("d-none");
            if (!liveClothingLinkPanel?.classList.contains("d-none")) {
                liveClothingUrl?.focus();
            }
        }));

        driveOpenTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            window.open("https://drive.google.com/drive/my-drive", "_blank", "noopener,noreferrer");
        }));

        storeOpenTriggers.forEach(trigger => trigger.addEventListener("click", () => {
            window.open("https://www.google.com/search?tbm=isch&q=clothing+product+front+view", "_blank", "noopener,noreferrer");
        }));

        const startCamera = async () => {
            if (stream) {
                cameraStatus.textContent = liveGarmentImage ? "Live try-on preview running" : "Live camera running";
                return;
            }

            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user",
                        width: { ideal: 640 },
                        height: { ideal: 960 },
                        frameRate: { ideal: 24, max: 30 }
                    }
                });
                video.srcObject = stream;
                cameraStatus.textContent = "Live camera running";
                setMode("camera");
                startLiveOverlay();
            } catch {
                window.alert("Camera access was denied by the browser.");
            }
        };

        form.querySelectorAll("[data-studio-tab]").forEach(tabButton => {
            tabButton.addEventListener("click", () => {
                const tab = tabButton.dataset.studioTab;
                if (!tab) {
                    return;
                }

                form.querySelectorAll("[data-studio-tab]").forEach(button => {
                    button.classList.toggle("is-active", button.dataset.studioTab === tab);
                });
                form.querySelectorAll("[data-studio-tab-panel]").forEach(panel => {
                    panel.classList.toggle("is-active", panel.dataset.studioTabPanel === tab);
                });

                submitRow?.classList.toggle("d-none", tab !== "files");

                if (tab === "live") {
                    startCamera();
                } else if (tab === "files") {
                    stopCamera();
                    cameraStatus.textContent = "Live camera ready";
                }
            });
        });

        form.querySelectorAll("[data-open-camera]").forEach(trigger => {
            trigger.addEventListener("click", startCamera);
        });

        form.querySelectorAll("[data-open-upload-page]").forEach(trigger => {
            trigger.addEventListener("click", () => {
                const uploadPageUrl = trigger.dataset.uploadPageUrl || "/TryOn/Upload";
                window.location.href = uploadPageUrl;
            });
        });

        fileCameraOpen?.addEventListener("click", async () => {
            fileCameraPanel?.classList.remove("d-none");

            if (fileCameraStream) {
                return;
            }

            try {
                fileCameraStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user",
                        width: { ideal: 960 },
                        height: { ideal: 1280 }
                    }
                });
                fileCameraFeed.srcObject = fileCameraStream;
            } catch {
                window.alert("Camera access was denied by the browser.");
            }
        });

        fileCameraCapture?.addEventListener("click", async () => {
            if (!fileCameraFeed?.videoWidth || !fileCameraFeed?.videoHeight || !fileCameraCanvas) {
                window.alert("Open the camera first, then capture the photo.");
                return;
            }

            const maxDimension = 800;
            const ratio = Math.min(maxDimension / fileCameraFeed.videoWidth, maxDimension / fileCameraFeed.videoHeight, 1);
            fileCameraCanvas.width = Math.max(1, Math.round(fileCameraFeed.videoWidth * ratio));
            fileCameraCanvas.height = Math.max(1, Math.round(fileCameraFeed.videoHeight * ratio));
            fileCameraCanvas.getContext("2d").drawImage(fileCameraFeed, 0, 0, fileCameraCanvas.width, fileCameraCanvas.height);
            imageData.value = fileCameraCanvas.toDataURL("image/jpeg", 0.82);
            poseUpload.value = "";
            if (poseUrl) {
                poseUrl.value = "";
            }
            setPreviewState("[data-pose-preview]", "[data-pose-empty]", imageData.value);
            await detectAndStorePose(imageData.value, posePreview, poseOverlayCanvas, poseLandmarksData);
            fileCameraPanel?.classList.add("d-none");
            stopFileCamera();
        });

        fileCameraClose?.addEventListener("click", () => {
            fileCameraPanel?.classList.add("d-none");
            stopFileCamera();
        });

        const captureCameraFrame = () => {
            if (!video.videoWidth || !video.videoHeight) {
                return false;
            }

            const maxDimension = 800;
            const ratio = Math.min(maxDimension / video.videoWidth, maxDimension / video.videoHeight, 1);
            canvas.width = Math.max(1, Math.round(video.videoWidth * ratio));
            canvas.height = Math.max(1, Math.round(video.videoHeight * ratio));
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            imageData.value = canvas.toDataURL("image/jpeg", 0.82);
            setPreviewState("[data-pose-preview]", "[data-pose-empty]", imageData.value);
            cameraStatus.textContent = "Frame captured";
            return true;
        };

        form.querySelector("[data-capture-photo]")?.addEventListener("click", () => {
            if (!captureCameraFrame()) {
                window.alert("Open the camera first, then capture the photo.");
            }
        });

        poseUpload?.addEventListener("change", async (event) => {
            imageData.value = "";
            clearPoseOverlay(poseOverlayCanvas, poseLandmarksData);
            const file = event.target.files?.[0];
            showPreview(file, "[data-pose-preview]", "[data-pose-empty]");

            if (!file) {
                return;
            }

            poseResizePromise = resizeImageFile(file, 720, 0.78)
                .then(dataUrl => {
                    imageData.value = dataUrl;
                })
                .catch(() => {
                    imageData.value = "";
                });

            await poseResizePromise;
            if (imageData.value) {
                await detectAndStorePose(imageData.value, posePreview, poseOverlayCanvas, poseLandmarksData);
            }
        });

        poseUrl?.addEventListener("input", async () => {
            imageData.value = "";
            clearPoseOverlay(poseOverlayCanvas, poseLandmarksData);
            poseUpload.value = "";
            const url = poseUrl.value.trim();

            if (!url) {
                setPreviewState("[data-pose-preview]", "[data-pose-empty]", "");
                return;
            }

            setPreviewState("[data-pose-preview]", "[data-pose-empty]", url);
            await detectAndStorePose(url, posePreview, poseOverlayCanvas, poseLandmarksData);
        });

        clothingUpload?.addEventListener("change", async (event) => {
            clothingImageData.value = "";
            const file = event.target.files?.[0];
            showPreview(file, "[data-clothing-preview]", "[data-clothing-empty]");
            loadLiveGarment(file ? URL.createObjectURL(file) : "");

            if (!file) {
                return;
            }

            clothingResizePromise = resizeImageFile(file, 640, 0.78)
                .then(dataUrl => {
                    clothingImageData.value = dataUrl;
                })
                .catch(() => {
                    clothingImageData.value = "";
                });

            await clothingResizePromise;
        });

        clothingUrl?.addEventListener("input", () => {
            clothingImageData.value = "";
            clothingUpload.value = "";
            const url = clothingUrl.value.trim();

            if (!url) {
                setPreviewState("[data-clothing-preview]", "[data-clothing-empty]", "");
                loadLiveGarment("");
                return;
            }

            setPreviewState("[data-clothing-preview]", "[data-clothing-empty]", url);
            loadLiveGarment(url, true);
        });

        liveClothingUrl?.addEventListener("input", () => {
            const url = liveClothingUrl.value.trim();
            if (clothingUrl) {
                clothingUrl.value = url;
            }
            if (clothingUpload) {
                clothingUpload.value = "";
            }
            clothingImageData.value = "";
            loadLiveGarment(url, true);
        });

        clothingCategory?.addEventListener("change", () => {
            smoothedFit = null;
            if (syncGarmentAreaFromCategory()) {
                startLiveOverlay();
            }
        });

        targetBody?.addEventListener("change", () => {
            smoothedFit = null;
            syncGarmentAreaFromCategory();
            startLiveOverlay();
        });

        garmentView?.addEventListener("change", () => {
            smoothedFit = null;
            startLiveOverlay();
        });

        liveAiFitTrigger?.addEventListener("click", async () => {
            syncGarmentAreaFromCategory();

            if (!captureCameraFrame()) {
                window.alert("Open the camera first, then try AI Fit.");
                return;
            }

            await Promise.allSettled([clothingResizePromise]);
            const hasClothing = Boolean(clothingImageData.value || clothingUpload?.files?.length || clothingUrl?.value.trim());

            if (!hasClothing) {
                window.alert("Choose a clothing image or link first.");
                return;
            }

            if (latestPoseLandmarks?.length && poseLandmarksData) {
                poseLandmarksData.value = serializePoseLandmarks(
                    latestPoseLandmarks,
                    video.videoWidth || canvas.width || 1,
                    video.videoHeight || canvas.height || 1);
            } else {
                await detectAndStorePose(imageData.value, posePreview, poseOverlayCanvas, poseLandmarksData);
            }

            isSubmittingLiveAiFit = true;
            liveAiFitTrigger.setAttribute("disabled", "disabled");
            submitButton?.setAttribute("disabled", "disabled");
            form.requestSubmit(submitButton || undefined);
        });

        form.addEventListener("submit", async (event) => {
            const submitterAction = event.submitter?.getAttribute("formaction") || "";
            if (submitterAction.includes("/TryOn/SaveProfile")) {
                return;
            }

            syncGarmentAreaFromCategory();

            const filesPanelIsActive = form.querySelector("[data-studio-tab-panel='files']")?.classList.contains("is-active");

            if (!filesPanelIsActive && !isSubmittingLiveAiFit) {
                event.preventDefault();
                window.alert("Open Files, choose a model image and clothing image, then generate the final try-on, or use AI Fit from Live.");
                return;
            }

            if (!isSubmittingAfterResize) {
                event.preventDefault();
                submitButton?.setAttribute("disabled", "disabled");

                await Promise.allSettled([poseResizePromise, clothingResizePromise]);

                const hasPose = Boolean(imageData.value || poseUpload?.files?.length || poseUrl?.value.trim());
                const hasClothing = Boolean(clothingImageData.value || clothingUpload?.files?.length || clothingUrl?.value.trim());

                if (!hasPose || !hasClothing) {
                    isSubmittingLiveAiFit = false;
                    liveAiFitTrigger?.removeAttribute("disabled");
                    submitButton?.removeAttribute("disabled");
                    window.alert("Choose both the model image and the clothing image first.");
                    return;
                }

                isSubmittingAfterResize = true;
                submitButton?.removeAttribute("disabled");
                form.requestSubmit(submitButton || undefined);
                return;
            }

            if (imageData.value) {
                poseUpload?.setAttribute("disabled", "disabled");
            }

            if (clothingImageData.value) {
                clothingUpload?.setAttribute("disabled", "disabled");
            }
        });

        window.addEventListener("beforeunload", () => {
            stopCamera();
            stopFileCamera();
        });

        setMode("camera");
        syncGarmentAreaFromCategory();

        if (initialClothingUrl) {
            if (liveClothingUrl) {
                liveClothingUrl.value = initialClothingUrl;
            }

            if (clothingUrl) {
                clothingUrl.value = initialClothingUrl;
            }

            setPreviewState("[data-clothing-preview]", "[data-clothing-empty]", initialClothingUrl);
            loadLiveGarment(initialClothingUrl, true);
        }

        if (isPublicLive || shouldAutoStartCamera) {
            startCamera();
        }
    }

    document.querySelectorAll("[data-upload-form]").forEach(uploadForm => {
        const targetBody = uploadForm.querySelector("[data-upload-target-body]");
        const category = uploadForm.querySelector("[data-upload-clothing-category]");
        const garmentArea = uploadForm.querySelector("[data-upload-garment-area]");
        const submitButton = uploadForm.querySelector("[data-upload-submit]");
        const modelCaptureInput = uploadForm.querySelector("[data-upload-model-capture]");
        const uploadCameraPanel = uploadForm.querySelector("[data-upload-camera-panel]");
        const uploadCameraFeed = uploadForm.querySelector("[data-upload-camera-feed]");
        const uploadCameraCanvas = uploadForm.querySelector("[data-upload-camera-canvas]");
        const uploadCameraOpen = uploadForm.querySelector("[data-upload-camera-open]");
        const uploadCameraCapture = uploadForm.querySelector("[data-upload-camera-capture]");
        const uploadCameraClose = uploadForm.querySelector("[data-upload-camera-close]");
        const uploadCameraCountdown = uploadForm.querySelector("[data-upload-camera-countdown]");
        const garmentFileInput = uploadForm.querySelector("[data-upload-garment-file]");
        const garmentFileOpen = uploadForm.querySelector("[data-upload-garment-file-open]");
        const uploadPoseLandmarks = uploadForm.querySelector("[data-upload-pose-landmarks]");
        const uploadPoseOverlay = uploadForm.querySelector("[data-upload-pose-overlay]");
        const uploadModelPreview = uploadForm.querySelector('[data-upload-preview="model"]');
        let uploadCameraStream = null;
        let uploadPosePromise = Promise.resolve("");
        let isSubmittingAfterUploadPose = false;
        let isUploadCountdownRunning = false;

        uploadForm.querySelectorAll("[data-upload-preview-input]").forEach(input => {
            input.addEventListener("change", () => {
                const previewName = input.dataset.uploadPreviewInput;
                const preview = uploadForm.querySelector(`[data-upload-preview="${previewName}"]`);
                const file = input.files?.[0];

                if (!preview) {
                    return;
                }

                if (!file) {
                    preview.classList.add("d-none");
                    preview.removeAttribute("src");
                    if (previewName === "model") {
                        clearPoseOverlay(uploadPoseOverlay, uploadPoseLandmarks);
                    }
                    return;
                }

                preview.src = URL.createObjectURL(file);
                preview.classList.remove("d-none");

                if (previewName === "model" && modelCaptureInput) {
                    modelCaptureInput.value = "";
                    uploadPosePromise = detectAndStorePose(preview.src, preview, uploadPoseOverlay, uploadPoseLandmarks);
                }
            });
        });

        garmentFileOpen?.addEventListener("click", () => {
            garmentFileInput?.click();
        });

        const stopUploadCamera = () => {
            uploadCameraCountdown?.classList.add("d-none");
            isUploadCountdownRunning = false;
            uploadCameraStream?.getTracks().forEach(track => track.stop());
            uploadCameraStream = null;
            if (uploadCameraFeed) {
                uploadCameraFeed.srcObject = null;
            }
        };

        const runUploadCountdown = async () => {
            if (!uploadCameraCountdown) {
                await new Promise(resolve => window.setTimeout(resolve, 2000));
                return;
            }

            uploadCameraCountdown.classList.remove("d-none");
            for (const value of ["2", "1"]) {
                uploadCameraCountdown.textContent = value;
                await new Promise(resolve => window.setTimeout(resolve, 1000));
            }
            uploadCameraCountdown.classList.add("d-none");
        };

        uploadCameraOpen?.addEventListener("click", async () => {
            if (!navigator.mediaDevices?.getUserMedia || !uploadCameraFeed) {
                window.alert("Camera is not available in this browser.");
                return;
            }

            try {
                uploadCameraStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user",
                        width: { ideal: 1280 },
                        height: { ideal: 1600 }
                    },
                    audio: false
                });
                uploadCameraFeed.srcObject = uploadCameraStream;
                uploadCameraPanel?.classList.remove("d-none");
            } catch {
                window.alert("Could not open the camera. Check browser camera permission.");
            }
        });

        uploadCameraCapture?.addEventListener("click", async () => {
            if (isUploadCountdownRunning) {
                return;
            }

            if (!uploadCameraFeed || !uploadCameraCanvas || !modelCaptureInput) {
                return;
            }

            if (!uploadCameraStream || !uploadCameraFeed.videoWidth || !uploadCameraFeed.videoHeight) {
                window.alert("Open the camera first, then capture the photo.");
                return;
            }

            isUploadCountdownRunning = true;
            uploadCameraCapture.setAttribute("disabled", "disabled");
            await runUploadCountdown();
            uploadCameraCapture.removeAttribute("disabled");
            isUploadCountdownRunning = false;

            if (!uploadCameraStream || !uploadCameraFeed.videoWidth || !uploadCameraFeed.videoHeight) {
                return;
            }

            const width = uploadCameraFeed.videoWidth || 900;
            const height = uploadCameraFeed.videoHeight || 1200;
            uploadCameraCanvas.width = width;
            uploadCameraCanvas.height = height;
            const context = uploadCameraCanvas.getContext("2d");
            context.drawImage(uploadCameraFeed, 0, 0, width, height);

            const dataUrl = uploadCameraCanvas.toDataURL("image/jpeg", 0.84);
            modelCaptureInput.value = dataUrl;

            const preview = uploadForm.querySelector('[data-upload-preview="model"]');
            const modelFileInput = uploadForm.querySelector('[data-upload-preview-input="model"]');
            if (preview) {
                preview.src = dataUrl;
                preview.classList.remove("d-none");
            }
            if (modelFileInput) {
                modelFileInput.value = "";
            }

            uploadPosePromise = detectAndStorePose(dataUrl, preview, uploadPoseOverlay, uploadPoseLandmarks);
            await uploadPosePromise;
            stopUploadCamera();
            uploadCameraPanel?.classList.add("d-none");
        });

        uploadCameraClose?.addEventListener("click", () => {
            stopUploadCamera();
            uploadCameraPanel?.classList.add("d-none");
        });

        category?.addEventListener("change", () => {
            syncCategoryCatalog(targetBody, category, garmentArea);
        });

        targetBody?.addEventListener("change", () => {
            syncCategoryCatalog(targetBody, category, garmentArea);
        });

        syncCategoryCatalog(targetBody, category, garmentArea);

        uploadForm.addEventListener("submit", async (event) => {
            syncCategoryCatalog(targetBody, category, garmentArea);

            if (!isSubmittingAfterUploadPose) {
                event.preventDefault();
                await uploadPosePromise;
                isSubmittingAfterUploadPose = true;
                uploadForm.requestSubmit(event.submitter || submitButton || undefined);
                return;
            }

            submitButton?.setAttribute("disabled", "disabled");
            if (submitButton) {
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" aria-hidden="true"></span> Generating...';
            }
        });
    });

    const historyModal = document.querySelector("[data-history-modal]");
    if (!historyModal) {
        return;
    }

    const modalImage = historyModal.querySelector("[data-history-modal-image]");
    const modalDate = historyModal.querySelector("[data-history-modal-date]");
    const modalDownload = historyModal.querySelector("[data-history-modal-download]");

    document.querySelectorAll(".history-image-button").forEach(button => {
        button.addEventListener("click", () => {
            modalImage.src = button.dataset.historyImage || "";
            modalDate.textContent = button.dataset.historyDate || "";
            modalDownload.href = button.dataset.historyDownload || "#";
            historyModal.classList.remove("d-none");
            document.body.classList.add("modal-open");
        });
    });

    historyModal.querySelectorAll("[data-history-close]").forEach(closeButton => {
        closeButton.addEventListener("click", () => {
            historyModal.classList.add("d-none");
            document.body.classList.remove("modal-open");
        });
    });
})();
