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

    const form = document.querySelector("[data-tryon-form]");

    if (form) {
        const sourceButtons = form.querySelectorAll("[data-source-trigger]");
        const cameraSection = form.querySelector("[data-camera-section]");
        const poseUploadSection = form.querySelector("[data-pose-upload-section]");
        const video = form.querySelector("[data-camera-feed]");
        const canvas = form.querySelector("[data-camera-canvas]");
        const imageData = form.querySelector("[data-image-data]");
        const clothingImageData = form.querySelector("[data-clothing-image-data]");
        const poseUpload = form.querySelector("[data-pose-upload]");
        const clothingUpload = form.querySelector("[data-clothing-upload]");
        const posePreview = form.querySelector("[data-pose-preview]");
        const poseEmpty = form.querySelector("[data-pose-empty]");
        const clothingPreview = form.querySelector("[data-clothing-preview]");
        const clothingEmpty = form.querySelector("[data-clothing-empty]");
        let stream = null;

        const resizeImageFile = (file, maxDimension = 1280, quality = 0.88) => new Promise((resolve, reject) => {
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

        const showPreview = (file, previewElement, emptyElement) => {
            if (!file) {
                previewElement.classList.add("d-none");
                emptyElement.classList.remove("d-none");
                previewElement.removeAttribute("src");
                return;
            }

            previewElement.src = URL.createObjectURL(file);
            previewElement.classList.remove("d-none");
            emptyElement.classList.add("d-none");
        };

        const stopCamera = () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
        };

        const setMode = (mode) => {
            const useCamera = mode === "camera";
            cameraSection.classList.toggle("d-none", !useCamera);
            poseUploadSection.classList.toggle("d-none", useCamera);

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

        form.querySelector("[data-open-camera]")?.addEventListener("click", async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                setMode("camera");
            } catch {
                window.alert("Camera access was denied by the browser.");
            }
        });

        form.querySelector("[data-capture-photo]")?.addEventListener("click", () => {
            if (!video.videoWidth || !video.videoHeight) {
                window.alert("Open the camera first, then capture the photo.");
                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext("2d").drawImage(video, 0, 0);
            imageData.value = canvas.toDataURL("image/jpeg", 0.88);
            posePreview.src = imageData.value;
            posePreview.classList.remove("d-none");
            poseEmpty.classList.add("d-none");
        });

        poseUpload?.addEventListener("change", async (event) => {
            imageData.value = "";
            const file = event.target.files?.[0];
            showPreview(file, posePreview, poseEmpty);

            if (!file) {
                return;
            }

            try {
                imageData.value = await resizeImageFile(file);
            } catch {
                imageData.value = "";
            }
        });

        clothingUpload?.addEventListener("change", async (event) => {
            clothingImageData.value = "";
            const file = event.target.files?.[0];
            showPreview(file, clothingPreview, clothingEmpty);

            if (!file) {
                return;
            }

            try {
                clothingImageData.value = await resizeImageFile(file);
            } catch {
                clothingImageData.value = "";
            }
        });

        form.addEventListener("submit", () => {
            if (imageData.value) {
                poseUpload?.setAttribute("disabled", "disabled");
            }

            if (clothingImageData.value) {
                clothingUpload?.setAttribute("disabled", "disabled");
            }
        });

        window.addEventListener("beforeunload", stopCamera);

        const prefersUploadByDefault =
            window.matchMedia("(max-width: 767.98px)").matches ||
            /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);

        setMode(prefersUploadByDefault ? "upload" : "camera");
    }

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
