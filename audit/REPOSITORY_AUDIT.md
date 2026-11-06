# ممیزی فنی مخزن Facial Age Estimation — PyTorch

تاریخ: ۲۰۲۶-۱۱-۰۶، منطقهٔ زمانی Asia/Tehran

مخزن: https://github.com/Ebimsv/Facial_Age_estimation_PyTorch

مبنای تاریخچه: `a2c3a4e7b57390b12c4b43e9c53b15a10ec557e4`

این سند snapshot ممیزی پیش از پیاده‌سازی است. پس از تأیید کاربر، P1.1 تا P1.6 اجرا شدند؛
وضعیت فعلی و شواهد اجرای آن در [ثبت پیشرفت نقشهٔ راه](IMPLEMENTATION_ROADMAP.md)
آمده است. یافته‌های تاریخی این سند به‌عنوان باگ‌های بازِ نسخهٔ جدید تفسیر نشوند.

## وضعیت جاری پس از P1.6 — ۲۰۲۶-۱۱-۰۶

| یافته | وضعیت جاری | شاهد و مرز تأیید |
|---|---|---|
| F21 | Complete در کد و mock؛ integration زنده unknown | downloader بدون installer، extra اختیاری، staging اختصاصی، archive دقیق، layout discovery/انتخاب صریح، validation و no-overwrite؛ ۴۴ تست fake API/ZIP. metadata زندهٔ Kaggle در دسترس نبود و archive واقعی دریافت نشد |
| F22 | Complete برای guardrailهای برنامه؛ تاریخچه/سرویس خارجی تأیید نشده | credential استاندارد/backup/token ignore و از بسته exclude می‌شود؛ canary مصنوعی و git check-ignore موفق‌اند. safe checkpoint در P1.3 موجود است؛ هیچ secret واقعی خوانده/rotate نشده و تاریخچهٔ پاک ادعا نمی‌شود |
| F06/F07 | سازگاری P1.5 حفظ شد | parser نام UTK مشترک شد؛ suite preparation و partition/hash موفق است. sanity check downloader جای full decode، provenance و identity review را نمی‌گیرد |
| F19 | Partial برای کل پروژه | ۲۶۸ تست کامل موفق و Ruff سبز؛ train/resume/end-to-end کامل و ONNX matrix همچنان باقی‌اند |

کامیت کد P1.6: `31216d5`. **۲۶۸ تست محلی موفق**؛ دو CUDA test روی RTX 5070 Laptop،
multiprocessing/spawn و ۴۴ دانلود آفلاین. پس از اصلاح نهایی پیام layout، ۴۴ تست downloader
دوباره موفق شدند. Python 3.13.9 / torch 2.9.0+cu128 در محیط موجود و بدون نصب dependency.
source Kaggle 2.2.4 فقط خوانده شد، نه import/auth؛ SDK واقعی/شبکه در تست‌ها استفاده نشد.
**CI کامیت `d821247` نیز موفق شد**: lint و Python 3.10/3.11/3.12؛
[شاهد ریموت](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/actions/runs/42271622415). CUDA بالا محلی است؛ CI پایه CPU است.

[گزارش یکپارچهٔ فاز اول](PHASE_1_IMPLEMENTATION_REPORT.md) شش گام، کامیت‌ها، تست‌ها و
دو فلوچارت را توضیح می‌دهد؛ [راهنمای دانلود](../docs/DOWNLOADING.md) حدود validation،
حفاظت مقصد و auth را روشن می‌کند. min ۱۸۰۰۰ نام معتبر و decode پنج نمونه، completeness
یا کیفیت کل دیتاست را تضمین نمی‌کند. publication فایل‌ها atomic نیست و credential ignore
اثبات نبود افشای گذشته نیست. CSV/وزن تاریخی و credential واقعی دست‌نخورده‌اند.

پیاده‌سازی فاز اول پایان یافت؛ گام بعد **P2.1** است. atomic checkpoint، resume، گروه
پارامترها، scheduler/AMP skip و lifecycle logging هنوز بازند؛ benchmark جدید نداریم.
بخش‌های پایین snapshot تاریخی‌اند.

## وضعیت تاریخی پس از P1.5 — ۲۰۲۶-۱۱-۰۶

| یافته | وضعیت جاری | شاهد و کار باقی‌مانده |
|---|---|---|
| F06 | Complete در پیاده‌سازی و تست مصنوعی | `src/data/validation.py`، `scan_images`، `stratified_split` و `UTKDataset` ستون/label/path/file/decode و strata ناممکن را بررسی می‌کنند؛ guard باقی‌ماندن همهٔ سن‌ها از حذف بی‌خطای کلاس جلوگیری می‌کند. اجرای آماده‌سازی واقعی تأیید نشده است |
| F07 | Complete برای protocol جدید؛ سابقه و leakage همچنان Partial | خروجی جدید محافظت‌شده، manifest source/version/seed/CSV+image hashes، reject/duplicate report و verification در loader؛ بدون annotation هویتی، near-duplicate و منشأ legacy هنوز unknown هستند |
| F14 | Partial | generator مستقل، worker RNG، strict algorithms اختیاری و محیط/Git/device/flags در `experiment_metadata` ثبت می‌شوند؛ fixed-environment repeatability تست شد. RNG restore/resume و محیط مرجع چندنسخه‌ای باقی‌اند |
| F28 | Partial | empty/zero-batch موجود حفظ شد؛ sparse age و partition خراب رد می‌شوند. edge-case عمومی metrics و ارزیابی مستقل split هنوز P2.2 هستند |
| F19 | Partial | suite به ۲۲۴ تست رسید؛ دادهٔ مصنوعی، tamper، repeat split، spawn، CUDA و checkpoint metadata پوشش دارند. train/resume end-to-end کامل همچنان باز است |

**۲۲۴ تست کامل موفق**؛ Ruff و whitespace موفق. همان Conda موجود با Python 3.13.9 /
torch 2.9.0+cu128، بدون نصب یا دانلود. suite کامل با دسترسی حافظهٔ اشتراکی اجرا شد و
GPU RTX 5070 Laptop قابل مشاهده بود؛ دو تست AMP/CUDA واقعاً موفق شدند. فرض scale در
تست قدیمی L1 اصلاح شد (۱۲۸ برای گرادیان قابل‌نمایش، سپس optimizer step)؛ scale یا
backoff خود Trainer تغییر نکرد. نتایج skip قبلی مربوط به sandbox محدود بودند؛ این شاهد
عدد/گرادیان CUDA است، نه آموزش GPU کامل یا تکرارپذیری بین سخت‌افزارها.
**CI کامیت `ba4cf39` نیز موفق شد**: lint و Python 3.10/3.11/3.12؛
[شاهد ریموت](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/actions/runs/42267465335). اجرای CUDA ذکرشده محلی است؛ runnerهای CI مبتنی بر CPU هستند.

چهار CSV tracked با hash یکسان باقی ماندند و partition تاریخی بدون تصاویر دوباره
بررسی شد. `utkface_age_stratified_v2` فقط روی دادهٔ مصنوعی تولید/تأیید شد؛ protocol واقعی
یا نتایج مدل تغییر نکردند. همهٔ exact-byte duplicateها پیش‌فرض از protocol جدید حذف
می‌شوند؛ skip hashing صریح و decode اختیاری‌اند. identity/near-duplicate و کیفیت برچسب
تأیید نشده‌اند؛ hash، احراز اصالت تولیدکنندهٔ manifest نیست.

راهنما: [DATA_PROTOCOL.md](../docs/DATA_PROTOCOL.md). manifest ناقص/تغییریافته رد می‌شود؛
بدون manifest، metadata تاریخی حدس زده نمی‌شود. ذخیرهٔ transactional پوشه، resume، guardrail
دانلود و benchmark هنوز باقی‌اند. گام بعدی P1.6 است. متن پایین snapshotهای قبلی است.

## وضعیت تاریخی پس از P1.4 — ۲۰۲۶-۱۱-۰۶

| یافته | وضعیت جاری | شاهد و مرز رفع |
|---|---|---|
| F04 | Complete | `AgeEstimationModel.forward` خام است؛ `L1AgeLoss` loss خام و decode محدود مدرن را جدا می‌کند. تست bias منفی/بالاتر از ۱۰۰ روی چهار CNN، گرادیان و کاهش loss را تأیید می‌کند |
| F08 | Complete برای نام‌گذاری و انتساب علمی؛ benchmark باقی است | `L1MeanVarianceLoss` و config پیش‌فرض `l1_mean_variance`، aliasهای سازگار و راهنمای `docs/LOSSES.md` تفاوت variant با joint loss مقاله را روشن می‌کنند. ادعای بهبود ثابت حذف شد؛ ablation در P4.1 است |
| F09 | Complete برای اصلاح عددی؛ اجرای CUDA تأیید نشده | softmax/moment/L1/decoder در float32 و variance مرکزی؛ `tests/test_losses.py` مقدار و گرادیان را با حالت‌های تحلیلی و مرجع float64 می‌سنجد؛ دو تست CUDA بدون GPU skip می‌شوند |
| F03 / F02 | رفتار inference حفظ شد | alias ذخیره‌شده و schema v1 حفظ می‌شوند؛ legacy خام و inference مدرن محدود می‌مانند؛ export raw با sidecar و تست decoder مشخص است |
| F23 / F24 | Partial | ادعاهای بهبود MAE، latency و confidence اصلاح شدند؛ جدول نتیجهٔ واقعی، protocol و مستندات جامع همچنان باقی‌اند |

نتیجهٔ محلی: **۱۷۹ تست جمع‌آوری‌شده، ۱۷۷ موفق، ۲ skip اختیاری CUDA؛ Ruff و whitespace
موفق**. Python 3.13.9 / torch 2.9.0+cu128، CPU، همان محیط موجود و بدون نصب dependency.
این نتیجه، آزمایش CUDA یا benchmark دقت/سرعت نیست. تست Gradio محلی اجرا شد؛ در CI
بدون extra مربوط، آن تست نیز می‌تواند skip شود. **CI کامیت `abd0304` موفق شد**:
lint و تست‌های Python 3.10/3.11/3.12؛ [شاهد ریموت](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/actions/runs/42237846889).

Trainer برای اجرای جدید objective revision=2 را در checkpoint و W&B ثبت می‌کند؛
`create_checkpoint` عمومی و تبدیل قدیمی به‌طور پیش‌فرض سابقهٔ آموزش نمی‌سازند.
محاسبهٔ float32 logits overflowشدهٔ مدل را ترمیم نمی‌کند؛ آزمون عددی برای ورودی finite است.
اثر اصلاح بر MAE واقعی unknown است و با protocol ثابت P4.1 باید سنجیده شود.
فایل‌های داده، وزن و credential تغییری نکردند. گام بعدی **P1.5** است؛ موارد
resume، hash/provenance داده و guardrail دانلود هنوز بازند. بخش‌های پایین سوابق تاریخی‌اند.

## وضعیت تاریخی پس از P1.3 — ۲۰۲۶-۱۱-۰۶

- نتیجهٔ محلی: **۱۵۹ تست موفق، Ruff و whitespace check موفق**؛ همان محیط کوندای
  قبلی، CPU، بدون نصب یا دانلود وزن/داده. ۱۸ تست checkpoint و ۵ تست pipeline اضافه شده‌اند؛
  packaging اکنون ۱۱ تست دارد، از جمله inference از wheel نصب‌شده خارج checkout.
- **F02: Complete در دامنهٔ قرارداد inference**؛ loader/transform/decoder مشترک و
  schema v1 در تمام entry pointها برقرار است؛ artifact بر تنظیمات محلی اولویت دارد.
- **F03: سازگاری legacy رفع شد؛ انتشار وزن جدید همچنان Partial**. مدل واقعی با
  head تاریخی و خروجی خام، زیر restricted loader و پرچم صریح اجرا شد. preprocessing از
  کد همراه اولین commit وزن (`432569f`) استخراج شده است، نه از حدس نام فایل.
- **F22: Partial**؛ unsafe fallback حذف شد؛ pickle قدیمی فقط با opt-in صریح است.
  فایل credential هنوز خارج کامیت و دست‌نخورده است و ignore آن در P1.6 باقی است.
- **F18: Partial**؛ export از قرارداد مشترک و sidecar استفاده می‌کند و parity
  TorchScript روی CNN آزموده شد؛ ONNX/runtime و ماتریس بیشتر هنوز P2.4 است.
- **F13: Partial**؛ تمام transformهای ارزیابی اکنون مشترک و قطعی‌اند، ولی train loader
  هنوز shuffle/drop_last دارد و ارزیابی به سه CSV وابسته است؛ این بخش در P2.2 باقی است.
- **F17/F19: Partial**؛ نمودار support سفارشی، device و تست‌های parity بهتر شده‌اند؛
  batch واقعی، webcam، decode/error UX کامل و resume/train/export matrix کامل نشده‌اند.
- F04/F08/F09 و resume/atomic save اصلاح نشده‌اند؛ اجرای P1.3 به معنی پایان فاز ۱ نیست.

وزن واقعی روی تصویر نمونه خروجی ۲۶٫۵ سال داد؛ این فقط اجرای smoke محلی است، نه نتیجهٔ
test-set یا شاهد MAE=4.73. فایل checkpoint و CSVها تغییر نکردند. راهنمای استفاده و
مهاجرت: [CHECKPOINTS.md](../docs/CHECKPOINTS.md). بخش P1.2 و ممیزی اولیهٔ پایین، سوابق تاریخی‌اند.

پیگیری ریموت: CI اولیهٔ P1.3 و P1.2 خطای تطبیق متن در تست interpolation مربوط به
normalization داشتند، نه پذیرش config نامعتبر. بررسی shape پیش از typed merge منتقل
شد تا پیام پروژه مستقل از زمان اعتبارسنجی داخلی OmegaConf باشد؛ به‌جای سست‌کردن تست
یا pin کتابخانه، همان guard و پیام حفظ شدند. تست پیام در OmegaConf 2.3.1 محیط دوم
نیز موفق بود. سپس **CI کامیت `44d7a37` با OmegaConf 2.4.0 در Pythonهای 3.10، 3.11 و
3.12 و lint موفق شد**؛ [شاهد ریموت](https://github.com/Ebimsv/Facial_Age_estimation_PyTorch/actions/runs/42233638033).
بنابراین شکست قبلی به‌عنوان رخداد تاریخی و رفع‌شده ثبت می‌شود، نه وضعیت باز نسخهٔ فعلی.

## وضعیت تاریخی پس از P1.2 — ۲۰۲۶-۱۱-۰۶

این بخش وضعیت فعلی را از مشاهدات تاریخی زیر جدا می‌کند؛ شرح اولیه برای حفظ شواهد
نسخهٔ مبنا باقی مانده است. نتیجهٔ محلی جدید: **۱۳۵ تست موفق و Ruff سبز** در محیط
کوندای موجود با Python 3.13.9 / torch 2.9.0+cu128؛ تست‌ها CPU و offline بودند.

| یافته | وضعیت جاری | تغییر تأییدشده و کار باقی‌مانده |
|---|---|---|
| F01 | Complete برای اصلاح بسته‌بندی | در P1.1 wheel/sdist و CLI نصب‌شده تأیید شد؛ حداقل runtime matrix همچنان F15 است |
| F05 | Complete در دامنهٔ schema/تنظیمات | schema typed، merge نهایی، immutable config، خطای typo، قرارداد mode/support، مهاجرت کلیدهای بی‌اثر و config preparation در `src/utils/config.py` و `src/configs` |
| F02 | Partial | ناسازگاری decoder/output اکنون fail-fast است؛ بازیابی مشترک تمام metadata در P1.3 باقی است |
| F06 | Partial | سن‌های finite/support و flags آماده‌سازی بررسی می‌شوند؛ decode، path، label کامل و sparse strata هنوز P1.5 هستند |
| F07 | Partial | پارامترها و تعداد واقعی در preparation_manifest ثبت می‌شود؛ hash/version/identity و provenance کامل هنوز وجود ندارد |
| F10 | Partial | اندازهٔ ViT از config می‌آید و 224/280 آزموده شد؛ heuristic گروه پارامترهای fc هنوز در P2.1 باقی است |
| F11 / F14 / F20 | Partial | config به‌صورت nested plain/resolved ذخیره می‌شود؛ resume، atomic save، RNG و lifecycle/logging کامل هنوز باقی است |
| F19 | Partial | ۸۸ تست config/contract و ۱۰ packaging موجودند؛ train/resume/export end-to-end هنوز کامل نیست |
| F28 | Partial | split خالی، train صفر batch و warmup با طول صفر پیش از مدل رد می‌شوند؛ edge-case عمومی metrics و ارزیابی مستقل split هنوز باقی است |

شواهد جدید: `tests/test_config.py`، `tests/test_packaging.py`، `scripts/train.py::main`،
`scripts/prepare_data.py::main`، `AgeEstimationModel.__init__`، guardهای loss و
`UTKDataset`/`build_dataloaders`. نحوهٔ تست و رفتار جدید در
[راهنمای تنظیمات](../docs/CONFIGURATION.md) و [ثبت پیشرفت](IMPLEMENTATION_ROADMAP.md) آمده است.

هیچ نتیجهٔ MAE یا benchmark جدیدی تولید نشده است. CSVهای موجود، وزن legacy و
`kaggle.json` تغییر نکرده‌اند. فرمول mean-variance و clamp رگرسیون اصلاح نشده‌اند؛
F03/F04/F08/F09 را به‌اشتباه بسته تلقی نکنید. بخش‌های «تست اجرا نشده» و «۳۷ تست»
در متن اولیه، مربوط به زمان ممیزی پیش از یافتن محیط‌های کوندا هستند.

## ۱. دامنه، روش و محدودیت شواهد

این ممیزی بر فایل‌های checkout محلی، ۷۲ فایل tracked، محتوای کامل `Readme.md`، راهنمای مشارکت، مجوز، تنظیمات، تمام فایل‌های پایتون tracked، workflow، CSVها و تاریخچهٔ مرتبط استوار است. صفحهٔ عمومی GitHub نیز مشاهده شد؛ یکسان‌بودن HEAD محلی با آخرین remote با fetch تأیید نشده است. مرجع پیاده‌سازی در این گزارش، checkout محلی است.

در مرحلهٔ تحلیل هیچ فایل مخزن تغییر نکرد و هیچ وابستگی نصب نشد. پس از تأیید کاربر فقط سه سند در `audit/` ایجاد شدند. هنگام شروع نگارش، `pyproject.toml` از قبل تغییر محلی داشت: ایمیل خالی نویسنده اصلاح شده بود. آن تغییر متعلق به این ممیزی نیست. یافتهٔ اولیهٔ ایمیل نامعتبر در نسخهٔ committed، برای working tree فعلی بسته شده است؛ سایر ایرادهای بسته‌بندی باقی‌اند.

روش‌های بررسی شامل خواندن کد، AST parsing بدون تولید bytecode، شمارش CSV با کتابخانهٔ استاندارد، اعتبارسنجی TOML با setuptools موجود، بررسی مسیر backend و خواندن metadata آرشیو checkpoint با `zipfile` و `pickletools` بوده است. pickle اجرا یا deserialize نشد. تصاویر مستندات به‌عنوان اثبات benchmark استفاده نشده‌اند؛ PDFهای مرجع داخل `papers/` پیاده‌سازی پروژه نیستند و بررسی خط‌به‌خط تمام آن مقالات در دامنهٔ این ممیزی نبوده است.

محدودیت‌های اجرایی:

- `dataset/` خالی است؛ صحت decode تصاویر، تکرار محتوایی و هم‌پوشانی هویت قابل بررسی نبود.
- در Python فعال، torch، torchvision، timm، numpy، pandas، OmegaConf، pytest، Ruff، TensorBoard، Gradio و ONNX نصب نیستند. وضعیت محیط‌های دیگر بررسی نشده است.
- آموزش، ارزیابی مدل، inference، تست‌های pytest، lint، build کامل wheel و export اجرا نشده‌اند. موفقیت CI آنلاین در این ممیزی تأیید نشده است.
- صحت نحوی ۲۴ فایل پایتون بررسی شد؛ این نتیجه جایگزین تست اجرایی نیست.
- هیچ MAE، latency، مصرف حافظه یا نتیجهٔ دیتاست جدیدی تولید نشده است.

## ۲. معماری و قابلیت‌های موجود

```text
UTKFace cropped images
  └─ scripts/prepare_data.py → master/train/valid/test CSV
       └─ src/data/UTKDataset + transform factories
            └─ scripts/train.py → OmegaConf → build_model + get_loss_fn
                 └─ Trainer → AdamW / OneCycleLR / AMP / logging / best checkpoint

checkpoint → evaluate.py | inference.py | demo/app.py | export_model.py
```

| بخش | پیاده‌سازی و کاربرد فعلی | ارزیابی |
|---|---|---|
| `src/data/` | dataset و سه کارخانهٔ transform، ساخت DataLoader | تفکیک مسئولیت مناسب؛ اعتبارسنجی محدود |
| `src/models/age_estimator.py` | MobileNetV3-Large، EfficientNet-B0/B3، ResNet50، ViT-S/14 DINOv2 | پنج شاخه در کد موجود؛ اجرای آنها اینجا تأیید نشده |
| `src/training/losses.py` | توزیع سنی و L1 regression | دو mode موجود؛ قرارداد و انتساب علمی نیازمند اصلاح |
| `src/training/trainer.py` | حلقهٔ train/validation، optimizer، scheduler، AMP، ذخیرهٔ بهترین مدل | پایهٔ مناسب؛ resume و ذخیرهٔ مقاوم ناقص |
| `scripts/` | دانلود، آماده‌سازی، آموزش، ارزیابی، تصویر/پوشه، export | مسیرهای اصلی وجود دارند؛ یکپارچگی checkpoint ناقص |
| `demo/app.py` | رابط Gradio و نمودار توزیع | موجود؛ مدل/تنظیمات و خطاهای ورودی نیازمند بهبود |
| `configs/default.yaml` | تنظیمات آموزش و داده | قابل override در train؛ بخشی از کلیدها بی‌اثر |
| `tests/` | تست model، loss، transforms و dataset مصنوعی | ۳۷ مورد ایستا؛ بخش‌های اصلی orchestration بدون پوشش |
| `.github/workflows/ci.yml` | Ruff و تست CPU روی Python 3.10/3.11/3.12 | موجود؛ نصب بسته و train/export را نمی‌آزماید |
| `notebooks/` | پوشهٔ محلی خالی؛ notebook tracked وجود ندارد | EDA آینده است |
| `papers/`, `pics/`, `figs/` | منابع و تصاویر آموزشی | مفید برای توضیح؛ اثبات نتیجهٔ آزمایش نیست |
| `checkpoints/`, `runs/` | وزن و eventهای قدیمی tracked | میراث نسخهٔ قبل؛ با ignore شدن از Git حذف نشده‌اند |
| `H-parameters.csv` | جدول متنی قدیمی با جداکننده‌های نمایشی | ثبت استاندارد آزمایش محسوب نمی‌شود |

کاربرد واقعی فعلی، آموزش و پژوهش تخمین سن روی چهرهٔ crop‌شدهٔ UTKFace است. adapter عمومی برای MORPH، FG-NET، CACD یا Adience وجود ندارد. ذکر آنها در README به معنی پشتیبانی اجرایی نیست. تشخیص چهره در عکس کامل، انتخاب چند چهره و tracking پیاده‌سازی نشده‌اند.

## ۳. کارهای درست و تکمیل‌شده در دامنهٔ مشخص

در این جدول Complete به معنی وجود پیاده‌سازی در دامنهٔ ذکرشده است، نه تضمین صحت همهٔ محیط‌ها.

| قابلیت | وضعیت | شاهد و دلیل |
|---|---|---|
| جداسازی داده/مدل/آموزش/utility | Complete | بسته‌های `src/data`, `src/models`, `src/training`, `src/utils`؛ برای اصلاحات تدریجی کافی است |
| رفع ignore اشتباه `src/data` | Complete | commit `de5e64d`، الگوهای `/data/` و `/dataset/` در `.gitignore` و tracked بودن هر سه فایل data |
| جلوگیری از ساخت مدل هنگام import | Complete | ساخت مدل در constructor/factory؛ `Trainer` در `src/training/__init__.py::__getattr__` lazy است |
| پنج backbone و head مشترک | Complete | `AgeEstimationModel.__init__`، `_make_head`؛ لازم نیست دوباره از صفر اضافه شوند |
| تبدیل RGB و target float | Complete | `UTKDataset.__getitem__`؛ تصویر `[3,H,W]` بعد از transform و target `[1]` |
| transform قطعی val/test | Complete | `get_val_transforms` و `get_test_transforms`؛ برخلاف docstring، center crop وجود ندارد و resize مستقیم است |
| bufferهای سن هم‌دستگاه با loss | Complete | `register_buffer` در `MeanVarianceLoss` و `.to(device)` در Trainer و predictorها |
| معیارهای پایهٔ سن | Complete | `compute_mae` و `compute_cumulative_score`؛ CS از شرط `<= threshold` استفاده می‌کند |
| تجمیع وزن‌دار batchها | Complete | `AverageMeter.update(val,n)` و استفاده با تعداد نمونه در Trainer |
| جداسازی train/eval mode | Complete | `model.train()` و `model.eval()`، همراه `no_grad` برای validation/inference |
| optimizer و clipping | Complete | AdamW، دو گروه LR و unscale پیش از clipping؛ اشکال تشخیص گروه ViT جداگانه ثبت شده |
| انتخاب checkpoint با validation | Complete | `Trainer.run` مقایسه با `val_mae`؛ test برای انتخاب مدل در حلقه استفاده نمی‌شود |
| logging اختیاری | Complete | TensorBoard و W&B با import هنگام فعال‌سازی؛ نیاز به بهبود lifecycle و metadata دارد |
| مجوز و مشارکت | Complete | `LICENSE` از نوع MIT و `CONTRIBUTING.md`؛ شرایط داده/وزن جداست |

## ۴. شواهد داده، تست و checkpoint

### ۴.۱. CSVهای tracked

| فایل | تعداد ردیف | بازهٔ سن مشاهده‌شده | نام تکراری داخل فایل |
|---|---:|---:|---:|
| `csv_dataset/utkface_dataset.csv` | 23,269 | 1–84 | 0 |
| `csv_dataset/train_set.csv` | 15,822 | 1–84 | 0 |
| `csv_dataset/valid_set.csv` | 2,793 | 1–84 | 0 |
| `csv_dataset/test_set.csv` | 4,654 | 1–84 | 0 |

اشتراک نام فایل بین هر جفت split صفر است. اجتماع رکوردهای split با master برابر است؛ سن CSV با سن رمزگذاری‌شده در نام فایل برای تمام رکوردهای master سازگار بود. نسبت‌ها تقریباً ۶۸٪/۱۲٪/۲۰٪ هستند؛ تنظیمات جدید ۸۰٪/۱۰٪/۱۰٪ است. این اختلاف، اثبات leakage نیست، اما مانع بازتولید یک آزمایش بدون تعیین split می‌شود. توزیع هویتی، کیفیت برچسب و تکرار تصویری نامعلوم است.

### ۴.۲. تست‌ها

شمارش ایستا با احتساب پارامترها: model=14، losses=10، dataset=4 و transforms=9؛ مجموع 37. `vit_small` اصلاً در `BACKBONES` تست نیست؛ گزینهٔ `-k "not vit_small"` این خلأ را رفع نمی‌کند. برای backboneهای CNN، `pretrained=False` استفاده شده است. تست فعلیِ clamp صرفاً رفتار موجود را تثبیت می‌کند و صحت گرادیان رگرسیون را نمی‌سنجد.

### ۴.۳. وزن قدیمی و منشأ 4.73

فایل `checkpoints/epoch-16-loss_valid-4.73.pt` برابر 96,458,132 بایت است. metadata آرشیو، یک OrderedDict از وزن‌ها با کلیدهای `model.fc.1.*` و `model.fc.2.*` نشان داد؛ قالب جدید به `model_state_dict` در یک wrapper نیاز دارد. تاریخچهٔ `00c6055^:model.py` head قدیمی ResNet را `2048 → 256 → 1` نشان می‌دهد؛ head جدید `2048 → 512 → GELU → output` است. صرف تغییر output_dim کافی نیست.

در `00c6055^:train.py`، مقدار استفاده‌شده در نام فایل از `best_loss` اعتبارسنجی و `nn.L1Loss` آمده است؛ بنابراین منشأ نام فایل validation است. مقدار واقعی test MAE این وزن در ممیزی تأیید نشده است. «۴٫۷۳ روی test» نباید نتیجهٔ اندازه‌گیری‌شده تلقی شود.

## ۵. دفتر یافته‌ها و توصیه‌ها

طبقه‌بندی: **Confirmed** یعنی کد/داده آن را اثبات می‌کند؛ **Improvement** یعنی بهبود موجه با اثر اندازه‌گیری‌نشده؛ **Potential** یعنی ریسک نیازمند آزمایش؛ **Optional** یعنی پیش‌نیاز انتشار پایه نیست. وضعیت‌ها: Complete، Partial، Missing، Needs Improvement. اندازهٔ Small یعنی تغییر محدود؛ Medium یعنی چند ماژول و تست؛ Large یعنی داده/آزمایش یا چرخهٔ انتشار. این اندازه‌ها تخمین کارند، نه زمان تضمین‌شده. اولویت Critical برای خطر فوری از‌دست‌رفتن/افشای داده محفوظ است؛ مسئلهٔ بحرانیِ بالفعل اثبات نشد، ولی چند High مانع انتشار قابل‌اعتماد هستند.

### F01 — بسته‌بندی و نصب واقعی

- وضعیت: Needs Improvement؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: `pyproject.toml:1–51`؛ backend=`setuptools.backends.legacy:build` در setuptools موجود قابل import نیست؛ schema کلید `namespace` را رد می‌کند؛ readme به `README.md` ناموجود اشاره دارد. `scripts/` و `demo/` فایل `__init__.py` ندارند، با discovery غیرnamespace مطلوب سازگار نیستند. entry pointها به `scripts.*` ارجاع دارند؛ YAML نیز به‌عنوان resource بسته تعریف نشده است.
- اهمیت: `pip install -r requirements.txt` نصب وابستگی است و خرابی نصب خود پروژه را آشکار نمی‌کند. اجرای entry point خارج از checkout قابل اتکا نیست.
- راه‌حل: backend رسمی `setuptools.build_meta`، `namespaces` صحیح، مسیر درست Readme، discovery صریح و resource تنظیمات؛ حفظ API فعلی تا حد ممکن. ایمیل working tree قبلاً اصلاح شده و کار باز نیست.
- پیش‌نیاز: ندارد؛ نصب wheel و اجرای CLI در محیط پاک باید بعداً آزموده شود.

### F02 — قرارداد یکسان checkpoint، preprocessing و decoder

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: `scripts/inference.py::load_model_and_pipeline`، `demo/app.py::load_model` و `scripts/export_model.py::main`، `ckpt['cfg']` را بازیابی نمی‌کنند؛ `scripts/evaluate.py::main` آن را merge می‌کند. در regression با YAML پیش‌فرض، decoder توزیعی ممکن است یک logit را روی ۱۰۱ سن broadcast کند و عدد غلط بسازد.
- اهمیت: مدل ظاهراً load می‌شود ولی اندازهٔ تصویر، normalization، support سن یا decoder می‌تواند اشتباه باشد.
- راه‌حل: loader مشترک، checkpoint version، config حل‌شدهٔ ساده، preprocessing و predictor مشترک؛ اعتبارسنجی سازگاری mode/output/support. تنظیمات معماری از artifact و overrideهای مجاز runtime صریح باشند.
- پیش‌نیاز: F01 برای استفاده از بستهٔ نصب‌شده؛ طراحی F03 و F05 هم‌زمان.

### F03 — وزن میراثی و شروع سریع با pretrained

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: checkpoint و تاریخچهٔ بخش ۴.۳؛ loaderهای فعلی `ckpt['model_state_dict']` می‌خواهند. در evaluation، نبود `val_mae` حتی می‌تواند قالب‌بندی `'?':.3f` را پیش از load خراب کند. وزن جدیدِ `best_*` tracked نیست و نتیجهٔ آموزش جدید موجود نیست.
- اهمیت: کاربر تازه بدون آموزش طولانی، مسیر تأییدشده برای اجرای وزن موجود ندارد؛ ادعای legacy mode دقیق نیست.
- راه‌حل: loader/adapter صریح legacy با معماری و preprocessing تاریخی، یا معرفی آن به‌عنوان artifact آرشیوی و انتشار وزن جدید پس از آموزش. استفاده از `strict=False` برای پنهان‌کردن اختلاف head راه‌حل نیست.
- پیش‌نیاز: F02؛ انتشار وزن جدید وابسته به F24 و F29 است.

### F04 — clamp در مسیر گرادیان رگرسیون

- وضعیت: Needs Improvement؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Small.
- شاهد: `AgeEstimationModel.forward:147–149`، `L1AgeLoss.predict_age` و `tests/test_model.py::test_regression_clamped`.
- اهمیت: clamp در خروجی خارج `[0,100]` مشتق صفر دارد و می‌تواند یادگیری رگرسیون را متوقف کند؛ وقوع و شدت آن در یک run اندازه‌گیری نشده است.
- راه‌حل: خروجی خام در forward آموزش؛ سیاست محدودکردن خروجی فقط در predictor و به‌صورت مستند. معیار ارزیابی باید روشن کند raw یا bounded prediction را می‌سنجد.
- پیش‌نیاز: قرارداد F02؛ تست گرادیان برای خروجی منفی و بالاتر از سقف.

### F05 — schema و اعتبارسنجی تنظیمات

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: `scripts/train.py::main` از dotlist آزاد استفاده می‌کند؛ `get_loss_fn` بررسی تطابق `output_dim == end_age-start_age+1` ندارد. `data.max_age/train_ratio/val_ratio` در train مصرف نمی‌شوند و prepare CLI مستقل است؛ `logging.save_best_only` و `inference.*` رفتار متناظر ندارند.
- اهمیت: typo یا ترکیب loss/output ناسازگار دیر یا با خروجی نادرست آشکار می‌شود؛ تغییر YAML الزاماً داده را تغییر نمی‌دهد.
- راه‌حل: structured config و fail-fast برای نام backbone، loss، age support، اندازه، ratios، batch، epochs و workers؛ کلیدهای بی‌اثر را وصل یا با migration روشن حذف کنید. config نهایی resolve و ثبت شود.
- پیش‌نیاز: ندارد؛ مبنای F02، F06 و F14 است.

### F06 — اعتبارسنجی ورودی داده و ساخت split

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: `create_master_csv` همهٔ ورودی‌های پوشه با نام قابل parse را می‌پذیرد؛ بررسی file/extension/decode و سن منفی ندارد. `stratified_split` فقط assert روی مجموع ratios دارد؛ sparse strata و ratios مرزی مدیریت نمی‌شوند. `UTKDataset.__init__` فقط نام ستون‌ها را بررسی می‌کند.
- اهمیت: فایل نامعتبر، NaN، target خارج support یا دادهٔ خالی می‌تواند دیرهنگام آموزش را خراب کند. stratification سن دقیق برای دیتاست‌های کوچک ممکن است ناممکن باشد؛ CSV فعلی این شکست را ثابت نمی‌کند.
- راه‌حل: schema رکورد، بازهٔ برچسب، path امن داخل root، گزارش نامعتبرها، بررسی اختیاری decode، خطای توضیحی ratios/strata. تغییر روش به age-bin فقط با نسخه و توضیح پروتکل انجام شود.
- پیش‌نیاز: F05؛ حفظ داده‌های معتبر و عدم حذف خاموش نمونه‌ها.

### F07 — منشأ split و حفاظت از test

- وضعیت: Partial؛ نوع: Confirmed برای اختلاف split، Potential برای leakage؛ اولویت: High؛ اندازه: Medium.
- شاهد: CSVها در بخش ۴.۱؛ `stratified_split` بر سن split می‌کند و identity/hash را نمی‌سنجد؛ master و split manifest ندارند.
- اهمیت: split جدید را نمی‌توان به نتایج قدیمی نسبت داد؛ نبود اشتراک نام، نبود leakage هویتی/تصویری را اثبات نمی‌کند.
- راه‌حل: حفظ CSV فعلی به‌عنوان legacy protocol، manifest شامل dataset source/version، فیلتر سن، seed، ratios، hash CSV و کد؛ بررسی hash محتوا و گزارش شباهت‌ها هنگام دسترسی به تصاویر. فقط در صورت داشتن identity معتبر از group split استفاده شود؛ هویت از timestamp حدس زده نشود. test برای انتخاب hyperparameter استفاده نشود.
- پیش‌نیاز: F06 و دسترسی مجاز به داده برای بررسی محتوا.

### F08 — تعریف و انتساب علمی loss

- وضعیت: Needs Improvement؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: `MeanVarianceLoss.forward` برابر L1(mean,target) به‌علاوهٔ variance است و CE ندارد؛ مقاله آموزش مشترک با softmax loss را شرح می‌دهد [S1]. docstring ادعای بهبود ثابت UTKFace و جلوگیری از پیش‌بینی غلطِ بیش‌ازحد مطمئن دارد؛ variance به‌تنهایی چنین تضمینی ندارد.
- اهمیت: استفاده از یک variant معتبر ممکن است، ولی معرفی آن به‌عنوان بازتولید دقیق مقاله و اثبات برتری علمی موجه نیست.
- راه‌حل: variant فعلی با نام و فرمول دقیق حفظ شود؛ اگر بازتولید مقاله هدف است، mode جدا مطابق معادلات و supervision مقاله اضافه شود. L1، variant فعلی و روش مرجع با split و budget یکسان مقایسه شوند.
- پیش‌نیاز: F05 و F07؛ آزمایش برتری وابسته به F24.

### F09 — پایداری عددی loss و predictor

- وضعیت: Needs Improvement؛ نوع: Potential؛ اولویت: Medium؛ اندازه: Small.
- شاهد: variance با تفریق `E[a²]-E[a]²` محاسبه می‌شود؛ `predict_age` softmax را خارج autocast و بدون cast صریح اجرا می‌کند. forward loss داخل autocast است؛ رفتار promotion موجود مانع ادعای overflow قطعی می‌شود.
- اهمیت: cancellation برای توزیع متمرکز و دقت پایین logits می‌تواند خطای کوچک بسازد؛ NaN یا variance منفی در run واقعی تأیید نشده است.
- راه‌حل: moment و decoder در float32، ترجیح variance مرکزی، تست concentrated/extreme logits، finite gradients و مقایسه با float64؛ clamp صرفاً خطای گردکردن کوچک را بپوشاند، نه مشکل بزرگ را.
- پیش‌نیاز: F08 برای مشخص‌بودن فرمول؛ محیط torch برای آزمون.

### F10 — گروه‌بندی پارامترهای ViT و اندازهٔ ورودی

- وضعیت: Needs Improvement؛ نوع: Confirmed برای heuristic و اندازه، نیازمند runtime verification برای فهرست دقیق پارامترها؛ اولویت: High؛ اندازه: Medium.
- شاهد: `Trainer._split_param_groups` از substringهای `head/classifier/fc` استفاده می‌کند؛ MLPهای timm دارای `fc1/fc2` هستند [S5] و می‌توانند LR سر مدل بگیرند. `AgeEstimationModel.__init__` اندازهٔ ViT را ثابت 224 قرار داده ولی transform از cfg.data.img_size می‌آید.
- اهمیت: differential LR با معنای مورد انتظار منطبق نیست و override عمومی img_size برای ViT قابل اتکا نیست.
- راه‌حل: ارجاع صریح به ماژول head و تقسیم بر شناسهٔ پارامتر؛ پوشش دقیق و بدون overlap. اندازهٔ ViT را از config بگیرید یا اندازه‌های مجاز را fail-fast محدود کنید؛ preprocessing checkpoint اصلی backbone بررسی و ثبت شود.
- پیش‌نیاز: F05؛ تست pretrained=False برای ViT بدون دانلود.

### F11 — resume و checkpoint مقاوم

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: `Trainer._save_checkpoint` optimizer را ذخیره می‌کند، اما scheduler/scaler/RNG/global_step و resume loader وجود ندارند. checkpoint قبلی قبل از `torch.save` حذف می‌شود؛ `dict(cfg)` تبدیل بازگشتی به container ساده نیست.
- اهمیت: قطع run قابل ادامهٔ دقیق نیست؛ خطای نوشتن می‌تواند بهترین checkpoint قبلی را از دسترس خارج کند.
- راه‌حل: `best` برای انتخاب مدل و `last` برای resume؛ نوشتن فایل موقت و atomic replace پیش از حذف نسخهٔ قبلی، schema version و config سادهٔ resolved. epoch/global_step، best metric، optimizer، scheduler، scaler و RNG ذخیره و بازیابی شوند؛ sampler state/workerها برای سطح تضمین resume تعیین شوند.
- پیش‌نیاز: F02، F05 و F14؛ تست failure injection ذخیره و restart.

### F12 — scheduler هنگام skip شدن optimizer با AMP

- وضعیت: Needs Improvement؛ نوع: Confirmed در کنترل جریان، Potential در رخداد overflow؛ اولویت: Medium؛ اندازه: Small.
- شاهد: `Trainer.train_one_epoch:162–166` scheduler را پس از هر scaler.step بدون بررسی انجام update جلو می‌برد.
- اهمیت: GradScaler می‌تواند optimizer update را skip کند؛ در آن صورت schedule و updateها همگام نمی‌مانند.
- راه‌حل: تعریف شمارندهٔ successful update و policy روشن برای OneCycleLR؛ skip و scale را log کنید و scheduler را مطابق آن جلو ببرید.
- پیش‌نیاز: F11 برای ذخیرهٔ شمارنده؛ تست شبیه‌سازی step skipped، بدون نیاز GPU دائمی.

### F13 — ارزیابی train و استقلال splitها

- وضعیت: Needs Improvement؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Small.
- شاهد: `evaluate.py::main` همان train_loader با augmentation/shuffle/drop_last را انتخاب می‌کند. `build_dataloaders` برای ارزیابی test نیز هر سه CSV را می‌خواند.
- اهمیت: train evaluation همهٔ نمونه‌ها را پوشش نمی‌دهد؛ test-only evaluation بی‌جهت به فایل‌های train/val وابسته است.
- راه‌حل: ساخت loader ارزیابی برای split منتخب با transform قطعی، shuffle=False و drop_last=False؛ شمارش n باید دقیقاً برابر dataset باشد.
- پیش‌نیاز: F02، F06 و F07؛ حفظ رفتار آموزش واقعی.

### F14 — بازتولیدپذیری فراتر از seed

- وضعیت: Partial؛ نوع: Improvement؛ اولویت: High؛ اندازه: Medium.
- شاهد: `set_seed` random/numpy/torch/CUDA و cuDNN را تنظیم می‌کند؛ نسخهٔ محیط، hash داده، worker generator و RNG resume ثبت نشده‌اند. `PYTHONHASHSEED` در فرایندِ از قبل شروع‌شده مقداردهی می‌شود و hash seed همان فرایند را عوض نمی‌کند.
- اهمیت: یک seed به‌تنهایی تضمین نتیجهٔ یکسان در همهٔ نسخه‌ها/دستگاه‌ها نیست. نبود worker_init_fn به‌تنهایی اثبات nondeterminism نیست؛ PyTorch خود seed workerها را مدیریت می‌کند [S4].
- راه‌حل: سطوح reproducibility تعریف شوند؛ ثبت محیط/Git/config/data hashes؛ generator و seed worker صریح در صورت RNGهای سفارشی، deterministic algorithms به‌صورت گزینهٔ کنترل‌شده و گزارش محدودیت‌ها. آزمایش تکرار در محیط ثابت انجام شود.
- پیش‌نیاز: F07 و F11؛ محیط مرجع آزموده‌شده.

### F15 — سازگاری حداقل وابستگی و دستگاه

- وضعیت: Needs Improvement؛ نوع: Confirmed برای حداقل torch، Improvement برای matrix؛ اولویت: High؛ اندازه: Medium.
- شاهد: وابستگی `torch>=2.0.0` ولی Trainer از `torch.amp.GradScaler` استفاده می‌کند؛ namespace نسخهٔ 2.0 فقط autocast صادر می‌کند [S3]. همهٔ inferenceها device خودکار می‌سازند و cfg.device را مانند train رعایت نمی‌کنند. وابستگی‌ها عمدتاً فقط حداقل دارند.
- اهمیت: نصب مجاز توسط metadata می‌تواند در import Trainer شکست بخورد؛ انتخاب CPU در سیستم GPUدار یکدست نیست.
- راه‌حل: حداقل واقعاً آزموده‌شده یا compatibility shim مشخص، جفت torch/torchvision معتبر، محیط مرجع versioned و CI برای حداقل/نسخهٔ مرجع. انتخاب device runtime مشترک و خطای روشن برای CUDA ناموجود؛ پشتیبانی MPS شرط انتشار پایه نیست.
- پیش‌نیاز: F01 و F05؛ بدون ارتقای کور همهٔ کتابخانه‌ها.

### F16 — خروجی ارزیابی قابل ردیابی

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: Medium؛ اندازه: Small.
- شاهد: `evaluate.py::main` در CSV نام تصویر ندارد و parent مسیر `--save-csv` را نمی‌سازد؛ نمونهٔ README از `results/predictions.csv` استفاده می‌کند. bias فقط میانگین گروه را چاپ می‌کند.
- اهمیت: خطای هر نمونه قابل پیگیری نیست؛ نبود پوشه خروجی فرمان مستند را خراب می‌کند؛ گروه کوچک می‌تواند گمراه‌کننده باشد.
- راه‌حل: شناسهٔ تصویر، target، prediction و error، metric JSON شامل n، hash protocol و checkpoint؛ ساخت parent؛ شمارش گروه‌ها، MAE و CS و اخطار برای n کم. bootstrap CI به‌عنوان بهبود مرحلهٔ benchmark، نه ادعای فعلی.
- پیش‌نیاز: F13 و F07؛ عدم انتشار تصاویر خام یا شناسه‌های حساس غیرضروری.

### F17 — کیفیت تجربهٔ inference و دمو

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: Medium؛ اندازه: Medium.
- شاهد: `inference.py::main` پوشه را تصویر‌به‌تصویر پردازش و فقط چاپ می‌کند؛ `annotate_image` parent نمی‌سازد. `demo.predict` ورودی None را مدیریت نمی‌کند؛ عنوان مدل در `build_interface` ثابت EfficientNet-B3 است؛ نمودار فقط support ثابت 0–100 دارد؛ webcam realtime و face detector موجود نیستند.
- اهمیت: batch folder موجود است، اما batching تانسوری و خروجی قابل‌مصرف ندارد؛ عکس کامل یا خالی می‌تواند نتیجهٔ نامعتبر یا exception بدهد.
- راه‌حل: مدل و support پویا، پیام ورودی خالی/خراب، مدیریت image orientation، خروجی CSV/JSON و batch-size قابل تنظیم؛ توضیح الزام single cropped face. detection فقط اگر نیاز عکس کامل تأیید شود اضافه شود.
- پیش‌نیاز: F02، F05؛ تست RGB/grayscale/RGBA، فایل خراب و پوشهٔ خالی.

### F18 — export و قرارداد deployment

- وضعیت: Partial؛ نوع: Confirmed برای نبود قرارداد/تست، نامعلوم برای اجرای موفق؛ اولویت: Medium؛ اندازه: Medium.
- شاهد: `export_model.py::main` trace و ONNX dynamic batch را تعریف کرده، اما فقط logits صادر می‌کند؛ img-size پیش‌فرض 224 و مستقل از artifact است. ONNX/ONNX Runtime در extras نیستند و checker/parity در CI نیست.
- اهمیت: فایل خروجی بدون preprocessing، age support و decoder برای مصرف‌کننده کافی نیست؛ موفقیت export به معنی برابری prediction نیست.
- راه‌حل: metadata همراه و در صورت نیاز wrapper expected-age، optional export extra متناسب با exporter نسخهٔ پشتیبانی‌شده؛ reload و parity با PyTorch روی B=1 و B>1. API مدرن export فقط پس از تعیین نسخهٔ مرجع انتخاب شود؛ تعویض ابزار به‌خودی‌خود هدف نیست.
- پیش‌نیاز: F02، F05، F15 و وزن معتبر؛ تست tolerance صریح.

### F19 — پوشش تست مسیر اصلی و CI

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Medium.
- شاهد: چهار فایل تست فعلی Trainer، metrics، prepare/download، CLI، reload/resume، inference/demo و export را پوشش نمی‌دهند؛ CI وابستگی‌ها را دستی نصب می‌کند و خود بسته را نصب نمی‌کند. TensorBoard در job تست نصب نمی‌شود و lazy import مانع دیده‌شدن آن در تست فعلی است.
- اهمیت: سبز بودن این suite صحت workflow انتهابه‌انتها یا packaging را تضمین نمی‌کند.
- راه‌حل: smoke test مصنوعی CPU با مدل کوچک برای train→save→reload→eval، unit testهای meaningful معیار/گرادیان/قرارداد، تست wheel در مسیر خارج repo و export اختیاری. حداقل یک تست ساخت واقعی هر backbone از جمله ViT بدون دانلود؛ GPU smoke در صورت زیرساخت، نه الزام همهٔ PRها.
- پیش‌نیاز: تست رگرسیون هم‌زمان با اصلاح هر F؛ گسترش CI پس از F01/F15.

### F20 — logging، lifecycle و تنظیمات بی‌اثر

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: Medium؛ اندازه: Small.
- شاهد: `Trainer.__init__/run/_log_scalar` یک log_dir عمومی دارد؛ close/finish در finally نیست؛ `config=dict(cfg)` برای W&B عمیق resolve نشده است. docstring `run` early stopping را ذکر می‌کند ولی منطق آن نیست؛ README bias را per-epoch می‌نامد ولی Trainer demographicها را دور می‌ریزد. step و epoch در W&B با یک step namespace ارسال می‌شوند.
- اهمیت: runها مخلوط و metadata مبهم می‌شوند؛ قدم‌های نامنظم W&B نیازمند تست اجرایی‌اند؛ قابلیت‌های ادعاشده با رفتار منطبق نیستند.
- راه‌حل: run-id/زیرپوشهٔ مستقل، resolved config، finally برای loggerها، محورهای روشن global_step/epoch و ثبت LR/skip/زمان. ادعاهای بی‌پشتوانه حذف شوند؛ early stopping اختیاری است و نیاز فعلیِ اثبات‌شده نیست.
- پیش‌نیاز: F05 و F11؛ dry-run loggerها با mock یا W&B offline.

### F21 — downloader قابل پیش‌بینی

- وضعیت: Needs Improvement؛ نوع: Confirmed؛ اولویت: Medium؛ اندازه: Small.
- شاهد: `download_from_kaggle` در صورت نبود Kaggle، pip install اجرا می‌کند، اولین zip پوشه را برمی‌دارد و پس از استخراج همان را حذف می‌کند؛ `verify_dataset` فقط پوشهٔ سطح اول و تعداد suffixها را می‌سنجد.
- اهمیت: تغییر پنهان محیط و انتخاب/حذف آرشیو نامرتبط ممکن است؛ معتبر بودن ۱۸هزار suffix به معنی سالم‌بودن تصاویر یا layout نیست. downloader در این ممیزی اجرا نشد.
- راه‌حل: Kaggle به‌عنوان extra اختیاری و خطای نصب روشن؛ پوشهٔ staging اختصاصی، نام آرشیو مورد انتظار و شناسایی layout، اعتبارسنجی فایل‌ها و انتقال پس از موفقیت؛ از حذف فایل غیرمتعلق به دانلود اجتناب شود.
- پیش‌نیاز: F06 و سیاست source/version در F07؛ تست با API mock و zipهای مصنوعی.

### F22 — credentials و بارگذاری artifact نامطمئن

- وضعیت: Needs Improvement؛ نوع: Confirmed برای رفتار، Potential برای سوءاستفاده؛ اولویت: High؛ اندازه: Small برای guardrails، Medium برای migration artifact.
- شاهد: `git status` یک `kaggle.json` untracked نشان داد؛ `.gitignore` آن را پوشش نمی‌دهد. محتوایش عمداً خوانده نشد و وجود secret واقعی یا افشای آن اثبات نشده است. loaderها `torch.load(...,weights_only=False)` دارند.
- اهمیت: commit تصادفی credential و اجرای pickle نامطمئن ریسک واقعی workflow است؛ وجود فایل untracked به‌تنهایی حادثهٔ امنیتی نیست.
- راه‌حل: ignore صریح credential و مستندسازی محل خارج repo؛ artifact ساده با metadata قابل JSON و safe weight loading در نسخهٔ پشتیبانی‌شده. legacy pickle فقط مسیر سازگاری صریح برای artifact مورد اعتماد؛ عدم fallback خاموش از safe به unsafe.
- پیش‌نیاز: F02/F03/F15؛ rotation فقط در صورت اثبات انتشار credential، نه به‌صورت خودکار.

### F23 — مستندات متناسب با پیاده‌سازی

- وضعیت: Needs Improvement؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Small.
- شاهد: `Readme.md` production-ready، سرعت‌های تقریبی بدون روش، بهبود قطعی loss، test حدود ۲۰۰۰ نمونه، پشتیبانی legacy، bias per-epoch و proper packaging را مطرح می‌کند؛ شواهد این گزارش خلاف یا ناکافی‌اند. wildcard checkpoint ممکن است صفر یا چند فایل match کند. `CONTRIBUTING.md` از `pre-commit install` می‌گوید ولی config tracked ندارد.
- اهمیت: شروع سریع شکست‌خورده و ادعاهای مبهم اعتماد جامعه را کاهش می‌دهد.
- راه‌حل: حذف/تعدیل ادعاهای تأییدنشده، نمونه‌خروجی برچسب‌خورده illustrative، فرمان با checkpoint مشخص، تفکیک implemented/verified/planned و دادهٔ supported/mentioned. دستور pre-commit تا ایجاد config اصلاح شود.
- پیش‌نیاز: F02/F03/F07 برای دستور نهایی؛ اصلاح ادعاهای نتیجه لازم نیست منتظر آموزش بماند.

### F24 — benchmark و آزمایش قابل‌بازتولید

- وضعیت: Missing برای نتایج مدل‌های جدید؛ نوع: Confirmed؛ اولویت: High؛ اندازه: Large.
- شاهد: جدول جدید MAE/CS برابر TBD است؛ فقط وزن و logهای قدیمی و `H-parameters.csv` وجود دارند. latencyهای جدول دستگاه، precision، batch، warmup و روش اندازه‌گیری ندارند.
- اهمیت: برتری مدل/loss و کاربرد عملی بدون protocol قابل سنجش نیست؛ زمان «آموزش در چند دقیقه» قابل تضمین نیست.
- راه‌حل: ابتدا یک baseline CNN و یک variant با budget یکسان؛ ثبت Git/config/env/split/checkpoint hashes، best validation و test مستقل، MAE/CS و تعداد گروه‌ها. latency روی سخت‌افزار مشخص با warmup/synchronization و تعریف preprocessing زمان‌گیری شود. چند seed برای ادعای مقایسه‌ای جدی‌تر، پس از baseline.
- پیش‌نیاز: F04–F16، F19 و دسترسی به داده/compute مجاز؛ بدون تولید عدد فرضی.

### F25 — مدل‌کارت، محدودیت‌ها و شرایط artifact

- وضعیت: Missing؛ نوع: Confirmed؛ اولویت: High پیش از انتشار وزن/دموی عمومی؛ اندازه: Medium.
- شاهد: `LICENSE` برای کد MIT است؛ model card و تفکیک حقوق dataset/backbone/checkpoint/تصاویر نمونه وجود ندارد. UTKFace در منبع رسمی استفادهٔ پژوهشی غیرتجاری را اعلام کرده است [S2].
- اهمیت: MIT کد مجوز یکسان برای همهٔ دارایی‌ها ایجاد نمی‌کند؛ پیش‌بینی سن یک تخمین با خطا و سوگیری است و خروجی softmax به‌خودی‌خود confidence کالیبره نیست.
- راه‌حل: منبع و شرایط هر asset، سن‌های واقعاً آموزش‌دیده (CSV فعلی 1–84)، crop/pose/domain، عملکرد گروه‌ها و کاربردهای خارج دامنه مستند شود. امکان انتشار وزن/دمو از شرایط مربوط بررسی شود؛ حکم حقوقی قطعی دربارهٔ وزن از این ممیزی استنتاج نشده است.
- پیش‌نیاز: F07/F08/F24؛ مستند محدودیت اولیه همین حالا قابل نوشتن است.

### F26 — بهداشت artifacts و منابع

- وضعیت: Needs Improvement؛ نوع: Improvement؛ اولویت: Low؛ اندازه: Small.
- شاهد: وزن ۹۶MB و سه TensorBoard event از قبل tracked هستند، با وجود ignore؛ PDFها و `H-parameters.csv` قدیمی در root/`papers/` قرار دارند.
- اهمیت: clone سنگین و ابهام provenance؛ حقوق بازتوزیع PDF/تصاویر باید روشن باشد.
- راه‌حل: inventory و checksum؛ وزن‌های آینده در Release/مدل‌رجیستری با شرایط مشخص، logهای منتخب با شرح منشأ، جدول آزمایش استاندارد. برای منابع در صورت نامشخص‌بودن حق بازتوزیع، لینک رسمی ترجیح داده شود. حذف تاریخچه یا فایل میراثی بدون تصمیم نگهداشت لازم نیست.
- پیش‌نیاز: F03/F25؛ هیچ بازنویسی Git history پیشنهاد ضروری نیست.

### F27 — تجربهٔ توسعه و مشارکت

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: Medium؛ اندازه: Small.
- شاهد: CONTRIBUTING و تنظیم Ruff موجود است؛ `.pre-commit-config.yaml`، Issue/PR template و فرایند release tracked نیستند. راهنما `ruff check --fix` را formatter معرفی می‌کند؛ formatting دستور مستقل دارد. annotation بازگشت `num_parameters` int است ولی float برمی‌گرداند.
- اهمیت: contributor نمی‌تواند تمام دستورهای راهنما را همان‌طور اجرا کند؛ اختلاف workflow محلی و CI اصطکاک ایجاد می‌کند.
- راه‌حل: دستورهای lint/format دقیق، config hooks سازگار با CI، PR template کوتاه و issueهای bug/feature، release checklist سبک و اصلاح annotation/docstringهای مشخص. refactor عمومی typeها اولویت ندارد.
- پیش‌نیاز: F19/F23؛ روال hooks پس از تثبیت tooling.

### F28 — کنترل خطای خالی و اندازهٔ batch

- وضعیت: Partial؛ نوع: Confirmed؛ اولویت: Medium؛ اندازه: Small.
- شاهد: `build_dataloaders` برای train با drop_last و n<batch می‌تواند صفر batch بسازد؛ scheduler صفر-step و `torch.cat([])` در validate محافظ ندارند. metrics ورودی خالی یا طول نامساوی را صریح رد نمی‌کنند.
- اهمیت: خطاهای نامفهوم و حتی میانگین بدون نمونه ممکن است؛ OneCycleLR الزاماً به drop_last=True نیاز ندارد.
- راه‌حل: fail-fast برای split خالی/صفر batch و شکل metric؛ drop_last را تصمیم آگاهانه برای training قرار دهید و مراحل scheduler را از loader واقعی محاسبه کنید. BatchNorm با batch بسیار کوچک نیز در smoke config در نظر گرفته شود.
- پیش‌نیاز: F05/F06؛ تست n کمتر/برابر/بیشتر batch و آخرین batch ناقص.

## ۶. قابلیت‌های اختیاری و بهبودهای مشروط

| شناسه / قابلیت | وضعیت و نوع | شاهد | ارزش و راه‌حل | پیش‌نیاز | اندازه / اولویت |
|---|---|---|---|---|---|
| O01: EDA قابل اجرا | Missing / Optional | `notebooks/` خالی؛ نمودارها موجودند | یک notebook/اسکریپت از CSV با توزیع سن، گروه‌ها و فیلتر؛ بدون خروجی ادعایی از run قدیم | F07 | Small / Low |
| O02: ONNX Runtime runner | Missing / Optional | فقط exporter وجود دارد | نمونهٔ CPU با قرارداد preprocessing و parity برای کاربران deployment | F18 | Medium / Medium |
| O03: detection و webcam realtime | Missing / Optional | estimator روی تصویر کامل resize می‌کند؛ loop و detector نیست | فقط برای کاربرد عکس کامل/وبکم، crop pipeline و سیاست zero/multi-face و fps واقعی | F17/F25 | Medium / Low |
| O04: CORAL یا روش مقایسه‌ای | Missing / Optional | مقاله ذکر شده ولی loss آن پیاده نشده | یک baseline ordinal معتبر پس از benchmark؛ ارزش مقایسه‌ای، نه افزایش تعداد گزینه‌ها | F08/F24 | Medium / Low |
| O05: GradCAM/attention | Missing / Optional | visualization توضیح‌پذیری وجود ندارد | نشان‌دادن نواحی حساس، همراه محدودیت تفسیر؛ تضمین توضیح علی نیست | F24 و مدل معتبر | Medium / Low |
| O06: cross-dataset evaluation | Missing / Optional | adapter سایر دیتاست‌ها نیست | سنجش domain shift با adapter، شرایط دسترسی و split هویتی معتبر | F07/F24/F25 | Large / Low |
| O07: دموی میزبانی‌شده | Missing / Optional | README آن را coming soon می‌نامد | مسیر اجرای کوتاه و وزن مشخص با محدودیت/حریم خصوصی روشن | F02/F17/F24/F25 | Medium / Medium |
| O08: بهینه‌سازی سرعت | Partial / Improvement | AMP و pinned transfer موجود؛ folder sequential و syncهای item در train | batching و profiling ابتدا؛ سپس channels-last/compile/quantization فقط با سود و parity اندازه‌گیری‌شده | F17/F18/F24 | Medium / Low |

DDP، Kubernetes، API service، hyperparameter platform یا چندین tracker جدید پیش‌نیاز این پروژه نیستند. RMS/RMSE، median error، signed error و macro group MAE می‌توانند گزارش را غنی کنند، اما MAE و CS معیارهای اصلی مناسب موجودند؛ افزودن metric صرفاً برای تعداد بیشتر توصیه نمی‌شود.

## ۷. تاریخچه و جلوگیری از تکرار کار

- `00c6055`: بازآرایی اصلی انجام شده؛ فایل‌های train/model/data جدید را نباید «مفقود» دانست.
- `95c63ed`: README جدید و جدول TBD؛ ادعاهای بازاریابی جای آزمایش را نگرفته‌اند.
- `de67cde` و `b50fbf6`: اصلاح lint و lazy import و CI؛ دوباره توصیه به ایجاد CI از صفر لازم نیست. افزودن train/install coverage کار باقی‌مانده است.
- `de5e64d`: مشکل tracked نبودن src/data حل شده است.
- `a2c3a4e`: downloader موجود است؛ بهبود آن لازم است، نه ایجاد دانلودکنندهٔ تکراری.

## ۸. موارد نامعلوم و روش تأیید

| موضوع | چرا نامعلوم است | راه تأیید در اجراهای بعدی |
|---|---|---|
| تست/CI فعلی سبز است؟ | ابزارهای محلی و نتیجهٔ workflow آنلاین در دسترس بررسی اجرایی نبود | اجرای suite در محیط مرجع و مشاهدهٔ run متناظر commit |
| MAE checkpoint قدیمی روی test | فقط منشأ validation نام فایل روشن است | legacy loader دقیق، preprocessing تاریخی و split ثابت، سپس test یک‌باره |
| بهبود loss یا backbone | جدول جدید TBD و ablation ثبت‌شده نیست | آزمایش کنترل‌شدهٔ F24 |
| leakage محتوایی/هویتی | تصاویر و identity manifest نداریم | hash و near-duplicate review؛ group split تنها با identity معتبر |
| حافظهٔ batch=96 روی GPU 8GB | benchmark حافظه وجود ندارد | peak allocated/reserved روی GPU، رزولوشن و نسخهٔ مشخص |
| parity TorchScript/ONNX | export و runtime اجرا نشدند | تست F18 روی چند batch و input با tolerance مشخص |
| صحت تعامل Gradio و W&B | extras نصب نیستند و تست ندارند | smoke test Gradio و offline logging در نسخهٔ مرجع |
| کیفیت برچسب و تعمیم | به تصاویر دسترسی نیست و cross-dataset نداریم | نمونه‌برداری کیفیت و ارزیابی domain جدا؛ طبق [S2] برچسب‌ها نیز محدودیت دارند |
| شرایط بازتوزیع وزن و assets | inventory حقوق هر asset موجود نیست | بررسی شرایط رسمی منبع داده/backbone و ثبت نتیجه، بدون تعمیم MIT |

## ۹. منابع و اسناد همراه

- [S1 — مقالهٔ اصلی Mean-Variance، CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.html): آموزش مشترک mean-variance و softmax؛ نتیجهٔ UTKFace این مخزن را اثبات نمی‌کند.
- [S2 — صفحهٔ رسمی UTKFace](https://susanqq.github.io/UTKFace/): دامنهٔ سن، برچسب‌ها، cropped faces و شرایط استفادهٔ داده.
- [S3 — namespace AMP در کد رسمی PyTorch 2.0](https://raw.githubusercontent.com/pytorch/pytorch/v2.0.0/torch/amp/__init__.py): فقط autocast صادر می‌شود؛ مبنای مشکل حداقل وابستگی.
- [S4 — راهنمای رسمی reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): برای اجرای بعدی و تعیین سطح تضمین بازتولیدپذیری؛ claim نتیجهٔ فعلی نیست.
- [S5 — MLP رسمی timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py): نام‌های fc1/fc2؛ فهرست دقیق نسخهٔ نصب‌شده باید در تست گروه‌ها تثبیت شود.
- [S6 — backend رسمی setuptools](https://setuptools.pypa.io/en/latest/build_meta.html): الگوی `setuptools.build_meta`.
- [نقشهٔ اجرای تدریجی](IMPLEMENTATION_ROADMAP.md)
- [بهبودهای متن‌باز و جامعه](OPEN_SOURCE_IMPROVEMENTS.md)

این سند snapshot ممیزی است. پس از هر PR وضعیت یافته و شواهد تست باید به‌روزرسانی شود؛ صرف ایجاد این اسناد به معنی اجرای توصیه‌ها نیست.
