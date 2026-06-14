(() => {
    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
    const config = {
        trackingIntervalMs: 55,
        poseModelComplexity: 1,
        drawArmsInFront: true,
        fallbackAfterMs: 4500,
        minLandmarkVisibility: 0.55,
        minShoulderRatio: 0.16,
        maxShoulderRatio: 0.74,
        minTorsoRatio: 0.18,
        maxTorsoRatio: 0.76
    };

    const normalizeCategory = (value) => {
        const normalized = (value || "")
            .trim()
            .toLowerCase()
            .replace(/[_\s]+/g, "-");

        return {
            "tee": "t-shirt",
            "tshirt": "t-shirt",
            "t-shirts": "t-shirt",
            "tanktop": "tank-top",
            "vest": "tank-top",
            "sleeveless": "tank-top",
            "sleeveless-shirt": "tank-top",
            "chemise-shirt": "chemise",
            "blouses": "blouse",
            "hoodies": "hoodie",
            "coats": "jacket",
            "trouser": "pants",
            "trousers": "pants",
            "jeans": "pants",
            "short": "shorts",
            "abaya": "abaya",
            "abayas": "abaya",
            "عباية": "abaya",
            "عبايات": "abaya",
            "galabiya": "galabeya",
            "jellabiya": "galabeya",
            "jalabiya": "galabeya",
            "overall": "jumpsuit",
            "overalls": "jumpsuit",
            "romper": "jumpsuit",
            "salopette": "jumpsuit",
            "salopeit": "jumpsuit",
            "سالوبيت": "jumpsuit"
        }[normalized] || normalized;
    };

    const areaDefaults = {
        upper: {
            area: "upper",
            anchor: "upper",
            widthFactor: 1.58,
            heightFactor: 1.30,
            centerFactor: 0.49,
            collarY: 0.13,
            neckLiftFactor: 0.03,
            sleeveLengthFactor: 0.96,
            sleeveWidthFactor: 0.20,
            sleeveShoulderOffset: 0.02,
            sleeveMode: "elbow",
            drawPoseSleeves: true,
            frontPanelOnly: true,
            frontCrop: { x: 0.18, y: 0.08, width: 0.64, height: 0.84 },
            backCrop: { x: 0.14, y: 0.07, width: 0.72, height: 0.86 },
            torsoTopWidthFactor: 1.20,
            torsoBottomWidthFactor: 1.02,
            torsoHeightFactor: 1.32,
            maxHeightFromShoulders: 2.08,
            maxWidthRatio: 0.82,
            maxHeightRatio: 0.74,
            minWidth: 84,
            minHeight: 104,
            opacity: 0.96
        },
        lower: {
            area: "lower",
            anchor: "lower",
            widthFactor: 1.18,
            heightFactor: 1.06,
            centerFactor: 0.52,
            maxWidthRatio: 0.58,
            maxHeightRatio: 0.70,
            minWidth: 78,
            minHeight: 128,
            waistY: 0.10,
            kneeY: 0.54,
            hemY: 0.98,
            opacity: 0.95
        },
        overall: {
            area: "overall",
            anchor: "overall",
            widthFactor: 1.34,
            heightFactor: 1.03,
            centerFactor: 0.53,
            maxWidthRatio: 0.74,
            maxHeightRatio: 0.86,
            minWidth: 98,
            minHeight: 190,
            shoulderY: 0.16,
            kneeY: 0.70,
            opacity: 0.95
        }
    };

    const categoryProfiles = {
        "t-shirt": {
            ...areaDefaults.upper,
            category: "t-shirt",
            widthFactor: 1.62,
            heightFactor: 1.12,
            centerFactor: 0.50,
            collarY: 0.145,
            neckLiftFactor: 0.035,
            sleeveLengthFactor: 0.78,
            sleeveWidthFactor: 0.19,
            sleeveShoulderOffset: 0.015,
            sleeveMode: "short",
            drawPoseSleeves: true,
            frontPanelOnly: true,
            frontCrop: { x: 0.16, y: 0.07, width: 0.68, height: 0.86 },
            backCrop: { x: 0.12, y: 0.06, width: 0.76, height: 0.88 },
            torsoTopWidthFactor: 1.16,
            torsoBottomWidthFactor: 0.96,
            torsoHeightFactor: 1.18,
            maxHeightFromShoulders: 1.72,
            maxWidthRatio: 0.78,
            maxHeightRatio: 0.62
        },
        chemise: {
            ...areaDefaults.upper,
            category: "chemise",
            widthFactor: 1.52,
            heightFactor: 1.20,
            centerFactor: 0.52,
            collarY: 0.15,
            neckLiftFactor: 0.035,
            sleeveMode: "long",
            sleeveLengthFactor: 0.90,
            sleeveWidthFactor: 0.19,
            drawPoseSleeves: true,
            frontPanelOnly: true,
            frontCrop: { x: 0.16, y: 0.07, width: 0.68, height: 0.88 },
            backCrop: { x: 0.13, y: 0.06, width: 0.74, height: 0.90 },
            torsoTopWidthFactor: 1.12,
            torsoBottomWidthFactor: 0.98,
            torsoHeightFactor: 1.24,
            maxHeightFromShoulders: 1.86,
            maxWidthRatio: 0.76,
            maxHeightRatio: 0.68
        },
        shirt: {
            ...areaDefaults.upper,
            category: "shirt",
            widthFactor: 1.50,
            heightFactor: 1.16,
            centerFactor: 0.51,
            collarY: 0.16,
            neckLiftFactor: 0.025,
            sleeveLengthFactor: 0.86,
            sleeveWidthFactor: 0.18,
            sleeveShoulderOffset: 0.015,
            sleeveMode: "long",
            drawPoseSleeves: true,
            frontPanelOnly: true,
            frontCrop: { x: 0.18, y: 0.08, width: 0.64, height: 0.84 },
            backCrop: { x: 0.14, y: 0.07, width: 0.72, height: 0.86 },
            torsoTopWidthFactor: 1.10,
            torsoBottomWidthFactor: 0.96,
            torsoHeightFactor: 1.22,
            maxHeightFromShoulders: 1.84,
            maxWidthRatio: 0.76,
            maxHeightRatio: 0.66
        },
        "tank-top": {
            ...areaDefaults.upper,
            category: "tank-top",
            widthFactor: 1.18,
            heightFactor: 1.06,
            centerFactor: 0.50,
            collarY: 0.16,
            neckLiftFactor: 0.02,
            sleeveMode: "none",
            drawPoseSleeves: false,
            frontPanelOnly: true,
            frontCrop: { x: 0.24, y: 0.07, width: 0.52, height: 0.86 },
            backCrop: { x: 0.20, y: 0.06, width: 0.60, height: 0.88 },
            torsoTopWidthFactor: 0.86,
            torsoBottomWidthFactor: 0.88,
            torsoHeightFactor: 1.16,
            maxHeightFromShoulders: 1.62,
            maxWidthRatio: 0.62,
            maxHeightRatio: 0.60
        },
        blouse: {
            ...areaDefaults.upper,
            category: "blouse",
            widthFactor: 1.54,
            heightFactor: 1.24,
            centerFactor: 0.52,
            collarY: 0.14,
            neckLiftFactor: 0.04,
            sleeveMode: "long",
            sleeveLengthFactor: 0.94,
            sleeveWidthFactor: 0.20,
            drawPoseSleeves: true,
            frontPanelOnly: true,
            frontCrop: { x: 0.16, y: 0.07, width: 0.68, height: 0.88 },
            backCrop: { x: 0.13, y: 0.06, width: 0.74, height: 0.90 },
            torsoTopWidthFactor: 1.14,
            torsoBottomWidthFactor: 1.00,
            torsoHeightFactor: 1.28,
            maxHeightFromShoulders: 1.90,
            maxWidthRatio: 0.76,
            maxHeightRatio: 0.70
        },
        hoodie: {
            ...areaDefaults.upper,
            category: "hoodie",
            widthFactor: 1.58,
            heightFactor: 1.22,
            centerFactor: 0.54,
            collarY: 0.15,
            neckLiftFactor: 0.08,
            sleeveLengthFactor: 0.88,
            sleeveWidthFactor: 0.22,
            sleeveShoulderOffset: 0.025,
            sleeveMode: "long",
            drawPoseSleeves: true,
            frontPanelOnly: true,
            frontCrop: { x: 0.16, y: 0.08, width: 0.68, height: 0.86 },
            backCrop: { x: 0.12, y: 0.07, width: 0.76, height: 0.88 },
            torsoTopWidthFactor: 1.18,
            torsoBottomWidthFactor: 1.04,
            torsoHeightFactor: 1.32,
            maxHeightFromShoulders: 1.86,
            maxWidthRatio: 0.72,
            maxHeightRatio: 0.66
        },
        jacket: {
            ...areaDefaults.upper,
            category: "jacket",
            widthFactor: 1.50,
            heightFactor: 1.24,
            centerFactor: 0.54,
            sleeveMode: "long",
            sleeveLengthFactor: 0.92,
            sleeveWidthFactor: 0.21,
            maxHeightFromShoulders: 1.88,
            maxWidthRatio: 0.74,
            maxHeightRatio: 0.68
        },
        pants: {
            ...areaDefaults.lower,
            category: "pants",
            widthFactor: 1.22,
            heightFactor: 1.08,
            centerFactor: 0.54,
            kneeY: 0.55,
            maxHeightRatio: 0.72
        },
        shorts: {
            ...areaDefaults.lower,
            category: "shorts",
            length: "short",
            widthFactor: 1.24,
            heightFactor: 0.72,
            centerFactor: 0.48,
            kneeY: 0.88,
            maxHeightRatio: 0.48
        },
        dress: {
            ...areaDefaults.overall,
            category: "dress",
            widthFactor: 1.36,
            heightFactor: 1.02,
            centerFactor: 0.54,
            kneeY: 0.68,
            maxWidthRatio: 0.76
        },
        galabeya: {
            ...areaDefaults.overall,
            category: "galabeya",
            widthFactor: 1.46,
            heightFactor: 1.04,
            centerFactor: 0.55,
            maxWidthRatio: 0.78,
            kneeY: 0.64,
            maxHeightRatio: 0.88
        },
        abaya: {
            ...areaDefaults.overall,
            category: "abaya",
            widthFactor: 1.50,
            heightFactor: 1.04,
            centerFactor: 0.55,
            collarY: 0.14,
            neckLiftFactor: 0.04,
            maxWidthRatio: 0.80,
            kneeY: 0.64,
            maxHeightRatio: 0.90,
            opacity: 0.96
        },
        jumpsuit: {
            ...areaDefaults.overall,
            category: "jumpsuit",
            widthFactor: 1.30,
            heightFactor: 1.05,
            centerFactor: 0.53,
            maxWidthRatio: 0.74,
            kneeY: 0.56,
            maxHeightRatio: 0.86
        }
    };

    const getProfile = (category, area) => {
        const normalizedCategory = normalizeCategory(category);
        const normalizedArea = (area || "").trim().toLowerCase();
        const profile = categoryProfiles[normalizedCategory];

        if (profile) {
            return profile;
        }

        return areaDefaults[normalizedArea] || areaDefaults.upper;
    };

    const getFactors = (area) => getProfile("", area);

    const clampAngle = (angle) => clamp(angle || 0, -0.22, 0.22);

    const isBrightBackground = (r, g, b, a) => {
        if (a < 8) {
            return true;
        }

        const brightest = Math.max(r, g, b);
        const darkest = Math.min(r, g, b);
        return (r > 244 && g > 244 && b > 244) ||
            (brightest > 214 && brightest - darkest < 44);
    };

    const prepareGarmentImage = (image) => {
        try {
            const width = image.naturalWidth || image.videoWidth || image.width;
            const height = image.naturalHeight || image.videoHeight || image.height;

            if (!width || !height) {
                return image;
            }

            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext("2d", { willReadFrequently: true });
            context.drawImage(image, 0, 0, width, height);

            const imageData = context.getImageData(0, 0, width, height);
            const data = imageData.data;
            const visited = new Uint8Array(width * height);
            const queue = [];
            let readIndex = 0;

            const tryQueue = (x, y) => {
                if (x < 0 || y < 0 || x >= width || y >= height) {
                    return;
                }

                const pointIndex = (y * width) + x;
                if (visited[pointIndex]) {
                    return;
                }

                const pixelIndex = pointIndex * 4;
                if (!isBrightBackground(
                    data[pixelIndex],
                    data[pixelIndex + 1],
                    data[pixelIndex + 2],
                    data[pixelIndex + 3])) {
                    return;
                }

                visited[pointIndex] = 1;
                queue.push(pointIndex);
            };

            for (let x = 0; x < width; x += 1) {
                tryQueue(x, 0);
                tryQueue(x, height - 1);
            }

            for (let y = 0; y < height; y += 1) {
                tryQueue(0, y);
                tryQueue(width - 1, y);
            }

            while (readIndex < queue.length) {
                const pointIndex = queue[readIndex];
                readIndex += 1;

                const x = pointIndex % width;
                const y = Math.floor(pointIndex / width);
                data[(pointIndex * 4) + 3] = 0;

                tryQueue(x + 1, y);
                tryQueue(x - 1, y);
                tryQueue(x, y + 1);
                tryQueue(x, y - 1);
            }

            let minX = width;
            let minY = height;
            let maxX = -1;
            let maxY = -1;

            for (let y = 0; y < height; y += 1) {
                for (let x = 0; x < width; x += 1) {
                    if (data[((y * width) + x) * 4 + 3] > 12) {
                        minX = Math.min(minX, x);
                        minY = Math.min(minY, y);
                        maxX = Math.max(maxX, x);
                        maxY = Math.max(maxY, y);
                    }
                }
            }

            if (maxX < minX || maxY < minY) {
                return image;
            }

            context.putImageData(imageData, 0, 0);

            const padding = Math.max(6, Math.round(Math.max(width, height) * 0.018));
            minX = Math.max(0, minX - padding);
            minY = Math.max(0, minY - padding);
            maxX = Math.min(width - 1, maxX + padding);
            maxY = Math.min(height - 1, maxY + padding);

            const cropped = document.createElement("canvas");
            cropped.width = maxX - minX + 1;
            cropped.height = maxY - minY + 1;
            cropped.getContext("2d").drawImage(
                canvas,
                minX,
                minY,
                cropped.width,
                cropped.height,
                0,
                0,
                cropped.width,
                cropped.height);

            return cropped;
        } catch (error) {
            console.warn("Could not prepare garment image", error);
            return image;
        }
    };

    const normalizeFit = (fit, displayWidth, displayHeight) => {
        if (!fit) {
            return fit;
        }

        const profile = fit.profile || getProfile(fit.category, fit.area);
        const maxWidth = displayWidth * (profile.maxWidthRatio || 0.64);
        const maxHeight = displayHeight * (profile.maxHeightRatio || 0.68);
        const yLimit = profile.area === "overall" ? 0.90 : profile.area === "lower" ? 0.86 : 0.78;

        return {
            ...fit,
            profile,
            angle: clampAngle(fit.angle),
            width: clamp(fit.width, profile.minWidth || 72, maxWidth),
            height: clamp(fit.height, profile.minHeight || 86, maxHeight),
            center: {
                x: clamp(fit.center.x, displayWidth * 0.18, displayWidth * 0.82),
                y: clamp(fit.center.y, displayHeight * 0.16, displayHeight * yLimit)
            }
        };
    };

    window.LiveFitAI = {
        clampAngle,
        getFactors,
        getProfile,
        normalizeCategory,
        normalizeFit,
        prepareGarmentImage,
        config
    };
})();
