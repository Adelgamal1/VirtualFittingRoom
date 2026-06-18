# تشغيل CatVTON على Kaggle GPU

استخدم الملف الجاهز:

`VirtualFittingRoom/kaggle_catvton_api.py`

## مهم قبل التشغيل

لو كنت شغلت خلية تثبيت عملت تحديث لـ `torch` أو `cuda-bindings` أو `numba-cuda` وظهرت conflicts كثيرة، اعمل من Kaggle:

`Session options -> Factory reset`

ثم شغل الخطوات التالية على Kernel نظيف.

## خلية التثبيت الآمنة

هذه الخلية لا تثبت `torch` ولا `cuda` ولا `numpy`، وتستخدم بيئة Kaggle الموجودة:

```python
import os, subprocess, sys
from pathlib import Path

WORK = Path('/kaggle/working')
CATVTON = WORK / 'CatVTON'

if not CATVTON.exists():
    subprocess.check_call([
        'git', 'clone', '--depth', '1',
        'https://github.com/Zheng-Chong/CatVTON',
        str(CATVTON)
    ])

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'fastapi', 'uvicorn', 'python-multipart',
    'fvcore', 'iopath', 'av'
])

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    'diffusers==0.29.2',
    'transformers==4.41.2',
    'accelerate==0.31.0',
    'safetensors==0.4.3',
    'huggingface_hub==0.23.4',
    'peft==0.11.1',
    'tokenizers==0.19.1'
])

print('Setup done')
```

## تشغيل الـ API

1. ارفع الملف `kaggle_catvton_api.py` إلى Kaggle.
2. شغل:

```python
!python /kaggle/working/kaggle_catvton_api.py --tunnel
```

3. انسخ الرابط الذي يطبع بهذا الشكل:

```text
https://xxxx.trycloudflare.com/tryon
```

4. الصقه في صفحة Upload داخل الموقع.
5. اترك خلية Kaggle شغالة طول ما تستخدم الموقع.

## لو ظهر نفس خطأ Cloudflare

رابط `trycloudflare.com` مؤقت. لو Kaggle وقف أو الخلية اتقفلت، شغل الأمر مرة أخرى وخذ رابط `/tryon` جديد.
