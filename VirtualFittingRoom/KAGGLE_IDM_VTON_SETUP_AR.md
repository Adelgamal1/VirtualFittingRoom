# تشغيل IDM-VTON على Kaggle GPU

الملف الجاهز:

`VirtualFittingRoom/kaggle_idm_vton_api.ipynb`

## الخطوات

1. افتح Kaggle.
2. اعمل Notebook جديد.
3. من Settings:
   - Accelerator: GPU T4 أو P100
   - Internet: On
4. ارفع ملف:
   `kaggle_idm_vton_api.ipynb`
5. Run All.
6. آخر خلية هتطبع رابط بهذا الشكل:

```text
https://xxxx.trycloudflare.com/tryon
```

7. حط الرابط في إعدادات المشروع كـ:

```powershell
setx VirtualTryOn__Mode "Api"
setx VirtualTryOn__ApiUrl "PASTE_KAGGLE_TRYON_URL_HERE"
setx VirtualTryOn__ApiPersonFieldName "person_image"
setx VirtualTryOn__ApiClothingFieldName "garment_image"
setx VirtualTryOn__ApiCategoryFieldName "category"
setx VirtualTryOn__ApiResponseImageField "outputImageBase64"
```

بعدها اقفل وافتح PowerShell أو Visual Studio، وشغل الموقع.

## ملاحظات مهمة

- لازم تسيب Notebook شغال طول ما بتستخدم الموقع.
- أول تشغيل بياخد وقت لأنه بينزل موديل IDM-VTON والـ checkpoints.
- لو Kaggle وقف الجلسة، شغل النوتبوك تاني وخد URL جديد.
- لو ظهر خطأ GPU أو CUDA، تأكد إن Accelerator مضبوط على GPU وليس CPU.
- بعد ما تحط رابط Kaggle الجديد، اقفل وشغل موقع ASP.NET من جديد.
- في وضع `Api` الموقع لن يرجع للـ overlay المحلي لو Kaggle فشل؛ سيظهر الخطأ بوضوح حتى تعرف أن الاتصال بـ Kaggle غير شغال.
- تأكد أن نوع القطعة المرسل من الموقع هو `t-shirt` أو `jersey/sports-shirt` للتيشيرتات والـ jerseys.
