# نقشهٔ راه اجرای تدریجی

تاریخ مبنا: ۲۰۲۶-۱۱-۰۶

مرجع یافته‌ها: [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md)

وضعیت: برنامهٔ مبنای ممیزی؛ پیشرفت اجرایی پس از تأیید کاربر در بخش زیر ثبت می‌شود.

## ثبت پیشرفت — P1.3

در ۲۰۲۶-۱۱-۰۶، P1.3 پیاده‌سازی و با **۱۵۹ تست محلی موفق** و Ruff تأیید شد.
گام بعدی **P1.4** است؛ مراحل بعدی به‌طور ضمنی انجام‌شده تلقی نمی‌شوند.

- `src/utils/checkpoint.py` قرارداد schema v1، ساخت artifact جدید، loader مشترک،
  load_state_dict سخت‌گیرانه، بارگذاری CPU اولیه و device/runtime override صریح دارد.
- model_spec، preprocessing و predictor با config ذخیره‌شده تطبیق داده می‌شوند؛
  metadata ناقص، نسخهٔ ناشناخته، خروجی ناسازگار و shape غلط وزن‌ها رد می‌شوند.
- مدل، transform و decoder یکسان در evaluation، CLI، Gradio و export استفاده می‌شوند.
  تغییر محلی defaults نمی‌تواند normalization/رزولوشن/support artifact را جایگزین کند.
  runtime YAML جزئی و داده/device قابل override هستند؛ contract سفارشی ناسازگار رد می‌شود.
  نسخهٔ دست‌نخوردهٔ YAML پیش‌فرض برای سازگاری قدیمی، fallback است و contract را override نمی‌کند.
- `src/inference/predictor.py` مسیر واحد تصویر/decoder است؛ نمودار دمو برای support
  سفارشی نیز درست است. export از اندازهٔ ذخیره‌شده استفاده می‌کند و sidecar JSON قرارداد
  logits می‌نویسد؛ این به معنی اتمام ماتریس deployment/ONNX در P2.4 نیست.
- Trainer artifact نسخه‌دار با config ساده تولید می‌کند؛ optimizer موجود حفظ می‌شود،
  ولی resume، scheduler/scaler/RNG و atomic save همچنان P2.1 هستند.
- raw state_dict فقط با `--legacy-resnet50` پذیرفته می‌شود. head و resize/normalization
  از commit `432569f` (اضافه‌شدن وزن واقعی) بازیابی شدند: 2048→256→1 بدون activation
  میانی، 128×128، normalization ImageNet و خروجی خام بدون clamp. تبدیل اختیاری به v1
  رفتار خام را حفظ می‌کند و فایل اصلی را خودکار تغییر نمی‌دهد.
- `weights_only=True` پیش‌فرض همهٔ loadهاست؛ retry نامطمئن خودکار وجود ندارد.
  wrapperهای قدیمی با DictConfig فقط با گزینهٔ صریح `--trust-legacy-pickle` برای فایل
  مورداعتماد پذیرفته می‌شوند. wrapper سادهٔ بدون نسخه با config کامل و هشدار مهاجرت load می‌شود.
- تست‌ها: ۱۸ checkpoint، ۵ inference/evaluation/demo/export، ۱۱ packaging، ۸۸ config
  و ۳۷ تست قبلی؛ Python 3.13.9 / torch 2.9.0+cu128، CPU، بدون نصب یا دانلود وابستگی/وزن/داده.
  مقایسهٔ Gradio در CI بدون extra مربوط skip می‌شود؛ محلی اجرا شد. pipeline نصب‌شدهٔ wheel
  و TorchScript parity نیز آزموده شدند. نتیجهٔ CI همین کامیت باید پس از push مشاهده شود.
- وزن واقعی `epoch-16-loss_valid-4.73.pt` با restricted loader و profile صریح روی CPU
  برای تصویر نمونه اجرا شد (خروجی گرد‌شده ۲۶٫۵ سال). این smoke test است، نه benchmark؛
  test MAE جدید یا ادعای برتری مدل تولید نشده است. هیچ CSV، وزن یا credential کاربر تغییر نکرد.

راهنمای فعلی: [docs/CHECKPOINTS.md](../docs/CHECKPOINTS.md).
F02 از نظر قرارداد و مسیر مشترک رفع شد؛ بخش legacy از F03 و بارگذاری محدود از F22 نیز
انجام شدند. انتشار وزن جدید/نتایج، محافظت credential، و بقیهٔ correctnessهای loss هنوز باقی‌اند.

پیگیری CI در همین مرحله: run اولیهٔ P1.3 و run قبلی P1.2 در تست پیام خطای
`data.mean=${model.name}` ناموفق بودند؛ خود ورودی رد می‌شد، ولی OmegaConf روی runner
زودتر پیام داخلی container-type صادر می‌کرد. shape normalization اکنون روی mapping
بدون type و پیش از merge schema بررسی می‌شود؛ تست سخت‌گیرانهٔ پیام حفظ شده است.
پیام mean/std در محیط دوم با OmegaConf 2.3.1 نیز جداگانه بررسی شد؛ آن بررسی تست کامل
محیط دوم نیست. نتیجهٔ CI کامیت اصلاحی باید با run همان کامیت تأیید شود.

## ثبت پیشرفت — P1.2

در ۲۰۲۶-۱۱-۰۶، گام P1.2 با تأیید کاربر پیاده‌سازی و محلی تأیید شد.
وضعیت آن مقطع: P1.1 و P1.2 انجام شده بودند؛ اکنون P1.3 نیز طبق ثبت بالا انجام شده است.
کامیت پیاده‌سازی: `6f738d6`؛ راهنمای استفاده و این ثبت پیشرفت در کامیت مستندات پس از آن آمده‌اند.

- `src/utils/config.py` schema مبتنی بر dataclass/OmegaConf، merge مشترک، خطای
  کلید ناشناخته، بررسی interpolation/type/مقادیر finite و خروجی resolved/read-only دارد.
  YAML جزئی و overrideهای CLI پیش از اعتبارسنجی نهایی با هم merge می‌شوند.
- سازگاری output/loss/support سن، اندازه و dropout مدل، normalization، نسبت‌ها، seed،
  batch/epoch/workers، LR/weight decay، clipping، warmup و قالب device بررسی می‌شوند.
- `train-age --print-config` و `prepare_data --print-config` بدون ساخت مدل یا دسترسی
  به تصاویر، config نهایی را نشان می‌دهند. خطای تنظیمات قبل از seed/data/model رخ می‌دهد.
- اندازهٔ ViT به timm منتقل می‌شود؛ forward واقعی بدون وزن pretrained در 224 و 280
  موفق بود؛ اندازهٔ 192 پیش از ساخت backbone رد می‌شود.
- lossها batch/shape/سن target را بررسی می‌کنند؛ broadcast اشتباه یک خروجی رگرسیون
  روی ۱۰۱ کلاس رد می‌شود. CSVهای train/eval از نظر finite و support سن پیش از image I/O
  بررسی می‌شوند؛ split خالی و train با صفر batch رد می‌شود.
- `prepare_data` از config مشترک و flagهای صریح با precedence مشخص استفاده می‌کند؛
  محدودهٔ سن را اعمال و `preparation_manifest.json` با پارامترهای مؤثر و تعدادها ذخیره می‌کند.
  تغییر ratios/max_age در train، CSVها را تغییر نمی‌دهد و الگوریتم split بازنویسی نشده است.
- `logging.save_best_only=true` و inference با مقادیر null با FutureWarning مهاجرت
  می‌کنند؛ مقدار false یا inference مؤثر/کلید ناشناخته با راهنمای صریح رد می‌شود.
- config ذخیره‌شده در checkpoint و W&B اکنون dict/list ساده و resolved است؛
  schema نسخه‌دار و بازیابی یکپارچهٔ artifact هنوز P1.3 است.
- **۱۳۵ تست کامل پاس شد**: ۸۸ config/contract، ۱۰ packaging و ۳۷ تست قبلی؛
  Python 3.13.9، torch 2.9.0+cu128، timm 1.0.28، CPU، بدون نصب وابستگی یا دانلود وزن/داده.
  Ruff و بررسی whitespace نیز موفق بودند. تست بسته‌بندی، wheel/sdist/rebuild و CLI
  معتبر/نامعتبر را خارج checkout می‌آزماید. step جدید این suite به CI اضافه شد؛ نتیجهٔ
  CI همین کامیت پس از push باید جداگانه مشاهده شود.

محدودیت‌ها: clamp رگرسیون و فرمول loss تغییر نکردند؛ metadata inference/export هنوز
به‌طور کامل restore نمی‌شود؛ hash داده، identity leakage، اعتبارسنجی decode و protocol
کامل هنوز در P1.5 هستند. manifest فعلی فقط ثبت پارامتر آماده‌سازی است. CSVهای tracked،
وزن‌ها و credential کاربر در این گام تغییر نکردند. راهنمای رفتار و مهاجرت:
[docs/CONFIGURATION.md](../docs/CONFIGURATION.md).

## ثبت پیشرفت — P1.1

در ۲۰۲۶-۱۱-۰۶، با تأیید کاربر و در محدودهٔ **فقط P1.1**، تغییرات نصب و بسته‌بندی
پیاده‌سازی شد. سایر گام‌های این سند هنوز برنامه‌اند و اجراشده تلقی نمی‌شوند.

- backend، discovery و metadata اصلاح شدند؛ ایمیل اصلاح‌شدهٔ کاربر حفظ شد.
- config پیش‌فرض در wheel قرار گرفت؛ copy سازگار checkout و `--config` صریح حفظ شدند.
- سه console command خارج checkout از wheel نصب‌شده کار کردند؛ مسیر نسبی داده در
  installed mode نسبت به cwd و در source mode نسبت به checkout حل می‌شود.
- wheel و sdist ساخته شدند؛ wheel مجدداً از sdist ساخته و در venv موقت نصب شد.
  هیچ‌یک از محیط‌های کوندا تغییر نکرد و dependency یا مدل/دیتاست دانلود نشد.
- **۴۵ تست کامل پاس شد** (۳۷ تست قبلی + ۸ تست packaging) در Python 3.13.9 و
  torch 2.9.0+cu128؛ تست‌ها روی CPU اجرا شدند. پس از آخرین تغییر manifest/docs،
  suite هشت‌تایی packaging نیز دوباره سبز شد. Ruff و `git diff --check` موفق بودند.
- محیط مستقل Python 3.11.0 با torch 2.11.0+cu128 نیز import Trainer و ساخت scaler
  غیرفعال را پاس کرد؛ این بررسی، تست کامل آن محیط نیست.
- حداقل API اعلامی torch 2.3 / torchvision 0.18 با source رسمی تطبیق داده شد؛
  runtime دقیق 2.3 موجود نبود و آزمایش نشد. job حداقل نسخه در P3.2 باقی است.
- build از خود checkout نیز موفق بود و محتوا برای نبود dataset، وزن، log، audit
  و credential بررسی شد. هشدارهای سازگاری نام `Readme.md` و metadata قدیمی license
  در راهنمای نصب توضیح داده شده‌اند؛ فایل README و LICENSE در artifact حاضرند.
- CI برای اجرای تست packaging به‌روز شد؛ سبزبودن CI روی GitHub نیازمند مشاهدهٔ run
  پس از push است و از تست محلی استنتاج نمی‌شود.

شرح محیط و دستورها: [docs/INSTALLATION.md](../docs/INSTALLATION.md).
ممیزی اصلی snapshot پیش از اجرا است؛ F01 در این گام رفع شد و F15 از نظر مرز API
اصلاح شد، اما matrix حداقل نسخه هنوز کامل نیست. پیشنهاد بعدیِ آن مقطع P1.2 بود؛
این گام اکنون طبق ثبت پیشرفت بالا انجام شده است.

## اصول اجرا و ترتیب وابستگی

هدف، رساندن مسیر موجود به یک workflow قابل نصب، صحیح و قابل‌بازتولید است. ساختار فعلی حفظ می‌شود؛ تغییر نام عمومی بسته یا بازنویسی framework لازم نیست. هر گام یک PR مستقل یا چند commit مرتبط داشته باشد. تست رگرسیون هر مشکل هم‌زمان با اصلاح آن نوشته شود؛ فاز سوم زمان شروع تست‌نویسی نیست.

نام فایل‌های جدید زیر پیشنهاد مشخص پیاده‌سازی است و وجود فعلی آنها ادعا نمی‌شود. تمام مسیرها نسبت به root مخزن‌اند. تغییر یا حذف فایل legacy فقط در گامی که صریحاً مشخص شده انجام شود؛ در این برنامه حذف فایل موجود الزامی نیست. پوشهٔ `audit/` محل نتایج آزمایش، تصاویر داده یا checkpoint جدید نیست.

| فاز | هدف و دروازهٔ خروج | اندازهٔ کلی |
|---|---|---|
| ۱ | قرارداد صحیح تنظیمات/داده/وزن و حذف موانع نصب و صحت؛ تست‌های هدفمند سبز | Medium، چند PR |
| ۲ | آموزش قابل ادامه و ارزیابی/inference/export یکپارچه؛ smoke انتهابه‌انتها | Medium، چند PR |
| ۳ | نصب واقعی و مسیر اصلی در CI، محیط مرجع و workflow مشارکت | Medium |
| ۴ | اجرای baseline واقعی و مستندات منطبق با نتیجه؛ quick start آزموده‌شده | Large به‌علت compute و داده |
| ۵ | انتشار artifact و قابلیت پیشرفته بر مبنای نیاز؛ بهبود جامعه | وابسته به انتخاب قابلیت |

ترتیب پیشنهادی گام‌ها:

```text
P1.1 → P1.2 → P1.3 → P1.4 → P1.5 → P1.6
                       └────────────→ P2.1 → P2.2 → P2.3 → P2.4
                                                      ↓
                                         P3.1 → P3.2 → P3.3
                                                      ↓
                                         P4.1 → P4.2 → P4.3
                                                      ↓
                                         P5.1 → P5.2 → P5.3
```

این نمودار ترتیب بازبینی است؛ پیش‌نیاز دقیق هر گام در متن آمده است. اصلاح ادعاهای نادرست README و جلوگیری از commit credential بهتر است زود انجام شود و لازم نیست منتظر فاز چهارم بماند. نصب dependency یا دانلود/آموزش در مرحلهٔ اجرای آینده باید با محدودیت‌های همان جلسه هماهنگ باشد؛ این نقشه مجوز نصب در ممیزی فعلی نیست.

## فاز ۱ — صحت، بازتولیدپذیری و قابلیت اعتماد

### P1.1 — رفع نصب پروژه و تعریف محیط پشتیبانی‌شده

- **هدف:** نصب dependency با نصب خود پروژه اشتباه نشود و حداقل نسخهٔ اعلام‌شده واقعاً قابل import باشد.
- **یافته‌ها / اولویت / اندازه:** F01، F15؛ High؛ Medium.
- **تغییر:** `pyproject.toml`، `requirements.txt` و `requirements-dev.txt`؛ حفظ ایمیل اصلاح‌شدهٔ working tree. در صورت حفظ entry pointهای فعلی، ایجاد `scripts/__init__.py` و `demo/__init__.py`؛ ایجاد `src/configs/__init__.py` و `src/configs/default.yaml` به‌عنوان resource بسته و اصلاح default-config lookup در entry pointها.
- **جزئیات:** backend صحیح، کلید `namespaces`، مسیر `Readme.md` و package discovery صریح؛ پکیج tests/data/artifacts در wheel وارد نشود. برای جلوگیری از دو منبع تنظیمات، `src/configs/default.yaml` منبع runtime باشد و `configs/default.yaml` فقط wrapper توسعه با دستور روشن یا فایل سازگاری کنترل‌شده باقی بماند؛ اگر duplication لازم شد consistency check بگذارید. حداقل torch را با APIهای مصرفی بسنجید؛ از حدس عدد نسخه پرهیز کنید. requirements را از metadata همسو نگه دارید.
- **پیش‌نیاز:** ندارد؛ محیط Python جدا برای تست نصب در اجرای آینده.
- **پذیرش:** sdist/wheel قابل ساخت و نصب؛ `train-age --help`، `eval-age --help` و `export-age --help` خارج از checkout کار کنند؛ default config داخل wheel پیدا شود؛ import Trainer در حداقل محیط منتخب موفق باشد.
- **تست:** ایجاد `tests/test_packaging.py` برای metadata/resource و یک CI smoke نصب artifact در P3.2؛ build artifacts در محل موقت. اثبات وجود فایل‌ها در wheel، نه صرف اجرای CLI از root.
- **خروجی:** پروژهٔ واقعاً نصب‌شدنی و مرز نسخه‌های قابل پشتیبانی؛ هنوز نتیجهٔ ML جدید ندارد.

### P1.2 — schema تنظیمات و قرارداد سن/مدل

**وضعیت اجرا: انجام‌شده در ۲۰۲۶-۱۱-۰۶؛ شواهد و مرز دامنه در ثبت پیشرفت P1.2.**

- **هدف:** ترکیب نامعتبر قبل از خواندن کل داده یا شروع GPU رد شود.
- **یافته‌ها / اولویت / اندازه:** F05، F10، F28؛ High؛ Medium.
- **ایجاد:** `src/utils/config.py`، `tests/test_config.py`.
- **تغییر:** `scripts/train.py`، `scripts/prepare_data.py`، `configs/default.yaml` و resource runtime منتخب در P1.1، `src/models/age_estimator.py`، `src/training/losses.py`.
- **جزئیات:** load/resolve/validate مشترک با structured config؛ خطای typo. regression الزاماً output_dim=1، توزیعی output_dim برابر طول support؛ start≤end، target داخل support و محدودیت max_age مستند. epochs/batch/ratios/std/size/workers بررسی شوند. اندازهٔ ViT از config گرفته یا برای گزینهٔ پشتیبانی‌نشده خطای روشن تولید شود. کلیدهای `save_best_only` و `inference.*` یا رفتار واقعی بگیرند یا با پیام migration کنار گذاشته شوند. آماده‌سازی داده باید پارامترهای مصرف‌شده را در manifest ثبت کند؛ train نباید وانمود کند تغییر ratio باعث بازسازی split می‌شود.
- **پیش‌نیاز:** P1.1.
- **پذیرش:** loss/model ناسازگار، typo و اندازهٔ نامعتبر ViT با پیام actionable رد شوند؛ config نهایی در container ساده و resolved قابل serialization باشد.
- **تست:** modeهای معتبر، تعداد class ناسازگار، سن منفی/خارج support، نسبت صفر/منفی/جمع نامعتبر، mean/std نامعتبر و صفر batch؛ تست config load از wheel و مسیر سفارشی.
- **خروجی:** یک تعریف قابل‌بررسی از تنظیمات و قرارداد tensor/age؛ CLI فعلی تا جای ممکن حفظ می‌شود.

### P1.3 — loader مشترک و سیاست legacy

**وضعیت اجرا: انجام‌شده در ۲۰۲۶-۱۱-۰۶؛ شواهد و محدودیت‌ها در ثبت پیشرفت P1.3.**

- **هدف:** artifact در evaluation، CLI و دمو همان preprocessing و age decoder آموزش را استفاده کند.
- **یافته‌ها / اولویت / اندازه:** F02، F03، F22؛ High؛ Medium.
- **ایجاد:** `src/utils/checkpoint.py`، `src/inference/__init__.py`، `src/inference/predictor.py`، `tests/test_checkpoint.py`، `tests/test_inference.py`.
- **تغییر:** `scripts/evaluate.py`، `scripts/inference.py`، `scripts/export_model.py`، `demo/app.py` و فرمت ذخیره در `src/training/trainer.py`.
- **جزئیات:** schema_version، model spec، age support، preprocessing spec و resolved config؛ loader model و decoder را فقط یک بار بسازد. metadata artifact از overrideهای runtime مانند device/data/output جدا شود؛ precedence صریح: defaults سپس artifact contract، سپس overrideهای مجاز runtime. model/output/loss ناسازگار را reject کنید. config ذخیره‌شده plain باشد تا safe load نسخهٔ مرجع ممکن شود.
- **legacy:** وزن فعلی را حذف نکنید. ابتدا معماری `2048→256→1` و preprocessing تاریخی را از history بازیابی کنید. loader صریح legacy یا ابزار تبدیل با flag مشخص بسازید؛ اگر preprocessing با اطمینان بازیابی نشد artifact آرشیوی معرفی شود و unknown ثبت شود. `strict=False` نباید جای مهاجرت معماری را بگیرد. از اجرای pickle نامطمئن و fallback خاموش به unsafe جلوگیری شود.
- **پیش‌نیاز:** P1.2؛ فایل وزن قدیمی فقط برای تست سازگاری محلی/اختیاری، نه هر job سبک CI.
- **پذیرش:** مدل با img_size/normalization غیرپیش‌فرض و regression با config پیش‌فرض خارجی، prediction یکسان قبل و بعد save/load بدهد؛ تمامی entry pointها این قرارداد را رعایت کنند. artifact unsupported خطای schema روشن بدهد.
- **تست:** round-trip هر دو output mode و support سفارشی؛ metadata ناقص، shape mismatch، safe-loading و conflict overrides؛ synthetic state dict قدیمی با همان head برای تست بدون دانلود. reload واقعی وزن میراثی در تست اختیاری.
- **خروجی:** یک منبع حقیقت برای inference و artifact قابل حمل بین ماشین‌ها.

### P1.4 — اصلاح گرادیان رگرسیون و تعریف دقیق loss

- **هدف:** مسیر آموزش قابل یادگیری و فرمول loss قابل دفاع باشد.
- **یافته‌ها / اولویت / اندازه:** F04، F08، F09؛ High؛ Medium.
- **تغییر:** `src/models/age_estimator.py`، `src/training/losses.py`، `tests/test_model.py`، `tests/test_losses.py` و توضیح loss در `Readme.md`.
- **جزئیات:** regression forward خام؛ bound در predictor به‌صورت policy مستند و یکسان. variant فعلی L1+variance با نام دقیق نگه داشته شود؛ برای بازتولید مقاله mode مجزا فقط پس از تطبیق کامل معادلات و classification loss ایجاد شود. moment/decoder float32 و variance مرکزی؛ ادعای confidence و بهبود ثابت حذف شود. تغییر فرمول loss، experiment identity جدید ایجاد کند.
- **پیش‌نیاز:** P1.2 و P1.3.
- **پذیرش:** برای خروجی خام منفی و بیش‌ازحد بزرگ، گرادیان مفید غیرصفر وجود داشته باشد؛ prediction policy مستقل از train باشد؛ loss و گرادیان finite و نزدیک مرجع تحلیلی باشند.
- **تست:** کنترل head با پارامتر مصنوعی برای خارج بازه؛ one-hot/uniform/two-point distribution، shift support، logits شدید و مقایسه float64؛ backward در حالت CPU و AMP اختیاری CUDA. تست قدیمی clamp به قرارداد predictor منتقل شود.
- **خروجی:** correctness رگرسیون و انتساب علمی روشن، بدون ادعای MAE بهتر پیش از آزمایش.

### P1.5 — دادهٔ معتبر، split ثابت و provenance

- **هدف:** هر run بداند دقیقاً با چه نمونه‌ها و فیلترهایی آموزش می‌بیند.
- **یافته‌ها / اولویت / اندازه:** F06، F07، F14، F28؛ High؛ Medium.
- **ایجاد:** `src/data/validation.py`، `tests/test_prepare_data.py`؛ در زمان آماده‌سازی `csv_dataset/split_manifest.json` و گزارش rejectها به‌عنوان خروجی تولیدی.
- **تغییر:** `scripts/prepare_data.py`، `src/data/dataset.py`، `src/utils/seed.py` و تست‌های dataset/config.
- **جزئیات:** فایل/پسوند/label/schema/مسیر معتبر؛ حالت decode-check اختیاری و گزارش علت حذف؛ stratification دقیق با بررسی حداقل نمونه و پیام برای حالت ناممکن. CSVهای legacy فعلی را بی‌صدا overwrite نکنید؛ protocol جدید در مسیر خروجی جدید ایجاد و hash شود. manifest شامل source/version، seed، max_age، ratios، تعدادها، hash لیست‌ها و Git revision باشد. hash محتوایی در صورت دسترسی به داده و گزارش تکرار؛ identity فقط با annotation معتبر. ثبت محیط و تعریف سطح determinism؛ generator و RNG سفارشی workerها کنترل شوند.
- **پیش‌نیاز:** P1.2 و تعریف artifact metadata در P1.3؛ برای بررسی تکرار تصویر، دادهٔ واقعی لازم است.
- **پذیرش:** ورودی و seed یکسان، CSV و manifest معنایی یکسان بدهند؛ train/val/test disjoint و اجتماع برابر master معتبر باشد؛ train خواندن دادهٔ خالی یا صفر batch را رد کند؛ protocol قدیم/جدید قابل تفکیک باشند.
- **تست:** دادهٔ مصنوعی با تعداد کافی برای هر stratum؛ sparse age، فایل غیرتصویر با نام UTK، invalid labels، schema ناقص، مسیر خارج root، empty set، repeat split و leakage نام فایل؛ تست hash تصویر هنگام دادهٔ مصنوعی موجود.
- **خروجی:** split مستند و قابل‌بازتولید؛ نبود leakage هویتی تا زمان تأیید داده همچنان unknown ثبت می‌شود.

### P1.6 — guardrail داده و دریافت قابل پیش‌بینی

- **هدف:** دانلود محیط را پنهان تغییر ندهد و credential/فایل نامرتبط وارد workflow نشود.
- **یافته‌ها / اولویت / اندازه:** F21، F22؛ High برای credential، Medium برای downloader؛ Small.
- **تغییر:** `.gitignore`، `scripts/download_data.py`، `pyproject.toml` و بخش دانلود `Readme.md`.
- **ایجاد:** `tests/test_download_data.py`؛ اضافه‌کردن Kaggle به extra اختیاری `data`.
- **جزئیات:** ignore برای credential، عدم خواندن یا چاپ secret؛ خارج‌کردن نصب خودکار pip از downloader، پیام نصب extra. دانلود در staging اختصاصی، انتخاب artifact دقیق، بررسی layout و محتوای حداقلی، cleanup فقط فایل متعلق به همین run؛ تشخیص layout واقعی Kaggle پیش از تثبیت مسیر.
- **پیش‌نیاز:** P1.5 برای validation؛ credential guard مستقل و زودتر قابل اجراست.
- **پذیرش:** بدون Kaggle پیام روشن و بدون تغییر محیط؛ وجود zip نامرتبط به انتخاب/حذف آن منجر نشود؛ `--check-only` هیچ دانلود/نصبی انجام ندهد؛ `git check-ignore` credential را match کند.
- **تست:** API mock، network failure، archive malformed، zip نامرتبط، layout nested، extraction failure و dataset ناقص؛ هیچ تماس واقعی Kaggle در CI پایه.
- **خروجی:** دریافت قابل پیش‌بینی؛ هیچ rotation خودکاری بدون شواهد افشا انجام نمی‌شود.

**دروازهٔ خروج فاز ۱:** نصب، config، round-trip، گرادیان، split مصنوعی و guardrailها آزموده شده‌اند؛ ادعاهای نادرست فوری در README اصلاح شده‌اند. حذف بهترین checkpoint پیش از ذخیرهٔ جدید باید در اولین PR فاز دوم برطرف شود؛ تا آن زمان آموزش طولانی جدید آغاز نشود.

## فاز ۲ — آموزش، ارزیابی و inference قابل استفاده

### P2.1 — آموزش قابل ادامه، ذخیرهٔ امن و logging مستقل

- **هدف:** قطع اجرا و restart، state آموزش و بهترین وزن را حفظ کند.
- **یافته‌ها / اولویت / اندازه:** F10، F11، F12، F14، F20؛ High؛ Medium.
- **تغییر:** `src/training/trainer.py`، `src/models/age_estimator.py`، `scripts/train.py` و config؛ utilityهای checkpoint/seed.
- **ایجاد:** `tests/test_trainer.py`، `tests/test_resume.py`.
- **جزئیات:** head module صریح با partition پارامترها بر مبنای identity؛ `best` و `last`، atomic save و retention بدون collision بین runها. ذخیرهٔ epoch/global_step/successful_updates، optimizer/scheduler/scaler/RNG/best_mae و protocol/environment؛ flag `--resume` و loader معتبر. تعریف resume تضمین‌شده از مرز epoch ابتدا؛ exact mid-epoch resume بدون sampler state ادعا نشود. schedule فقط با optimizer update موفق پیش برود؛ سیاست تعداد steps پس از skip و تغییر epochs روشن باشد. log_dir و save_dir زیر run-id، finally cleanup و محور W&B روشن. early stopping صرفاً در صورت نیاز بعداً اضافه شود.
- **پیش‌نیاز:** P1.3–P1.5؛ برای AMP واقعی GPU لازم است، تست skip می‌تواند mock باشد.
- **پذیرش:** failure نوشتن، best قبلی را سالم نگه دارد؛ اجرای دو run تداخل نکند؛ CPU run قطع‌شده در مرز epoch با run پیوسته در محیط قطعی به وزن/optimizer/LR یکسان در tolerance برسد؛ parameter groupها همهٔ پارامترها را دقیقاً یک بار پوشش دهند و ViT fc1/fc2 در backbone باشند.
- **تست:** train کوتاه با tiny network، loss کاهش‌یابنده و finite، atomic failure injection، round-trip optimizer/scheduler، deterministic restart، mock scaler skip و logger cleanup در exception؛ W&B offline smoke در extra job.
- **خروجی:** چرخهٔ آموزش قابل اعتماد و ادامه‌دادن؛ ادعای exact resume محدود به حالت واقعاً تست‌شده.

### P2.2 — ارزیابی قطعی و خروجی قابل پیگیری

- **هدف:** هر split به‌طور کامل و با protocol معلوم ارزیابی شود.
- **یافته‌ها / اولویت / اندازه:** F13، F16، F28؛ High؛ Small/Medium.
- **تغییر:** `scripts/evaluate.py`، `src/data/dataset.py`، `src/utils/metrics.py`؛ امکان بازگرداندن sample id بدون شکستن مصرف training فراهم شود.
- **ایجاد:** `tests/test_evaluate.py`، `tests/test_metrics.py`.
- **جزئیات:** loader فقط برای split منتخب، transform قطعی، drop_last=False؛ رعایت runtime device/data path پس از restore artifact؛ خروجی parent ساخته شود. CSV با image_name/true_age/predicted_age/error و metadata گروه‌ها؛ JSON با MAE/CS@5/CS@10، n و artifact/protocol hashes. group count و MAE/CS؛ RMSE و median/signed error فقط اگر به تحلیل کمک کنند.
- **پیش‌نیاز:** P1.3، P1.5؛ P2.1 برای artifact نهایی.
- **پذیرش:** n دقیقاً برابر طول split، حتی برای آخرین batch کوچک؛ train evaluation نیز ثابت؛ test evaluation بدون train CSV قابل اجرا؛ خروجی خطاها قابل اتصال به هر تصویر باشد.
- **تست:** predictions دستی با خطاهای 0/5/10 و بالای مرز؛ تجمیع batchهای نابرابر، split کوچک/خالی، تکرار اجرای قطعی، parent ناموجود و override runtime مسیر داده.
- **خروجی:** گزارش قابل‌بازتولید و قابل تحلیل، بدون تغییر پروتکل برای بهترکردن نتیجهٔ test.

### P2.3 — inference و دموی قابل اعتماد

- **هدف:** عکس/پوشهٔ ورودی به خروجی روشن و سازگار با evaluation برسد.
- **یافته‌ها / اولویت / اندازه:** F17؛ Medium؛ Medium.
- **تغییر:** `scripts/inference.py`، `demo/app.py` و `src/inference/predictor.py`.
- **ایجاد/گسترش تست:** `tests/test_inference.py` و `tests/test_demo.py` در extra suite.
- **جزئیات:** batching تانسوری اختیاری برای پوشه، CSV/JSON نتیجه و خطای نمونه، ترتیب ثابت، ساخت parent؛ None و فایل خراب با پیام روشن، EXIF orientation و RGB conversion. نام مدل و support نمودار از artifact؛ figureها پس از مصرف با lifecycle صحیح آزاد شوند. UI دامنهٔ single cropped face و محدودیت خروجی را توضیح دهد؛ تضمین confidence یا پشتیبانی تمام عکس‌ها ندهد.
- **پیش‌نیاز:** P1.3، P1.4 و P2.2 برای برابری گزارش.
- **پذیرش:** single/batch/demo برای ورودی و artifact یکسان prediction همسان در tolerance بدهند؛ تصویر خراب کل batch را بی‌دلیل متوقف نکند؛ regression و support غیر۱۰۱ نمودار مناسب داشته باشند.
- **تست:** grayscale/RGBA، تصویر چرخیده، empty folder، None، malformed file، batch=1 و batch>1 و output parent ناموجود؛ smoke load رابط با Gradio مرجع.
- **خروجی:** inference کاربردی و demo محلی معتبر؛ میزبانی عمومی هنوز وابسته به فاز پنج است.

### P2.4 — export قابل راستی‌آزمایی

- **هدف:** artifact export قرارداد ورودی/خروجی و parity عددی داشته باشد.
- **یافته‌ها / اولویت / اندازه:** F18؛ Medium؛ Medium.
- **تغییر:** `scripts/export_model.py`، `pyproject.toml`؛ استفاده از predictor spec مشترک.
- **ایجاد:** `tests/test_export.py` و در صورت نیاز `src/inference/export.py`.
- **جزئیات:** اندازه و normalization از artifact؛ مشخص‌کردن logits یا age output با metadata؛ dynamic batch طبق قابلیت واقعی runtime. extra `export` شامل ONNX/exporter dependencies نسخهٔ انتخاب‌شده و ONNX Runtime برای verification؛ خروجی parent ساخته شود. TorchScript trace فعلی در صورت parity حفظ شود؛ انتخاب مسیر جدید export تصمیم نسخه‌محور باشد.
- **پیش‌نیاز:** P1.3، P1.4، P1.1/P1.2 برای version support؛ P2.3 برای predictor واحد.
- **پذیرش:** reload TorchScript و ONNX checker/runtime روی batchهای 1 و 3، prediction نزدیک PyTorch؛ max absolute difference و tolerance در تست ثبت شوند. backbone unsupported صریح fail کند و عمومی advertise نشود.
- **تست:** CNN سبک واقعی pretrained=False، هر دو mode، input اندازهٔ ذخیره‌شده؛ ViT به‌عنوان تست کند/اختیاری تا توان CI مشخص شود. عدم نصب extras خطای راهنما بدهد.
- **خروجی:** deployment sample معتبر با دامنهٔ پشتیبانی مشخص.

## فاز ۳ — تست، کیفیت کد و نگهداشت

### P3.1 — suite معنی‌دار و سطوح تست

- **هدف:** تست‌ها خطاهای واقعی pipeline را بگیرند و به دانلود داده/وزن وابسته نباشند.
- **یافته‌ها / اولویت / اندازه:** F19، F27؛ High؛ Medium.
- **تغییر:** `tests/test_model.py`، تمام تست‌های افزوده‌شده، pytest config در `pyproject.toml`؛ ایجاد `tests/conftest.py` برای fixture مشترک و markerها.
- **جزئیات:** سه سطح fast unit، CPU integration و optional extras/GPU؛ ساخت ViT pretrained=False بدون دانلود در suite مناسب. metric edge case، fixture دادهٔ کافی برای stratification و overfit tiny sample در حد smoke؛ هدف numerical invariant و رفتار کاربر است، نه تست خط‌به‌خط implementation. warningهای مرتبط با compatibility به‌صورت blanket پنهان نشوند.
- **پیش‌نیاز:** اصلاحات فازهای ۱ و ۲، که تست خود را قبلاً دارند.
- **پذیرش:** اجرای offline بخش پایه، پوشش train→checkpoint→reload→eval، markerهای مستند و زمان اجرای مناسب؛ تعداد تست به‌عنوان هدف کیفیت استفاده نشود.
- **تست:** اجرای کامل suite در محیط مرجع، کنترل اینکه skipها فقط به extra/hardware مربوط باشند؛ coverage برای کشف مسیر فاقد تست، نه رسیدن مصنوعی به درصد خاص.
- **خروجی:** regression suite قابل نگهداشت.

### P3.2 — CI نصب و محیط مرجع

- **هدف:** نصب وابستگی دستی نتواند packaging شکسته یا dependency ناقص را مخفی کند.
- **یافته‌ها / اولویت / اندازه:** F01، F15، F19؛ High؛ Medium.
- **تغییر:** `.github/workflows/ci.yml`، dependency specs؛ ایجاد `requirements-tested.txt` یا constraint file مستند متناسب با روش منتخب.
- **جزئیات:** lint و unit matrix پایتونِ واقعاً پشتیبانی‌شده؛ wheel build، install خارج از repo و CLI smoke، integration CPU کوچک؛ extras export در job جدا. محیط مرجع reproducible و یک job حداقل نسخهٔ ادعاشده؛ job latest اختیاری برای هشدار compatibility. dependencyها از بسته/extras نصب شوند؛ torch CPU wheel از source مناسب. CI permission حداقلی و pin tooling هماهنگ با dev.
- **پیش‌نیاز:** P1.1، P1.6 و P3.1؛ نسخهٔ دقیق toolها هنگام پیاده‌سازی از منبع رسمی بررسی شود.
- **پذیرش:** checkout تمیز بدون dataset/credential/external network برای تست‌ها؛ wheel شامل کد و config لازم و فاقد وزن/secret باشد؛ نصب از checkout تنها مسیر موفقیت نباشد.
- **تست:** workflow run متناظر PR و artifact inventory؛ ثبت نسخه‌ها و traceback خوانا. GPU job فقط در صورت زیرساخت مجاز.
- **خروجی:** نشان CI با معنای فنی روشن.

### P3.3 — مشارکت و tooling یکدست

- **هدف:** contributor همان بررسی‌های CI را با دستورهای کوتاه اجرا کند.
- **یافته‌ها / اولویت / اندازه:** F27؛ Medium؛ Small.
- **ایجاد:** `.pre-commit-config.yaml`، `.github/ISSUE_TEMPLATE/bug_report.yml`، `.github/ISSUE_TEMPLATE/feature_request.yml`، `.github/pull_request_template.md`.
- **تغییر:** `CONTRIBUTING.md`، تنظیم Ruff و annotation/docstringهای مشخص در `src/models/age_estimator.py`، `src/data/transforms.py` و `src/training/trainer.py`.
- **جزئیات:** تمایز `ruff check` و `ruff format`، hooks سازگار با CI و دامنهٔ demo؛ دستور CPU بدون الزام CUDA؛ template شامل command/env/traceback بدون درخواست secret. مسیر افزودن backbone با تست head groups و config، نه صرف یک elif.
- **پیش‌نیاز:** P3.2 برای تثبیت دستورها.
- **پذیرش:** راهنمای مشارکت روی checkout تمیز قابل دنبال‌کردن؛ template مختصر و actionable؛ تغییر سراسری formatting بدون ضرورت در PR اصلاح صحت مخلوط نشود.
- **تست:** ruff check و format check مطابق policy، validation ساختار template و اجرای دستورهای راهنما؛ برای متن خالص تست unit جدید لازم نیست.
- **خروجی:** ورودی روشن برای مشارکت با هزینهٔ نگهداشت کم.

## فاز ۴ — مستندات، تجربهٔ کاربر و شواهد واقعی

### P4.1 — baseline و نتیجهٔ قابل‌بازتولید

- **هدف:** حداقل یک مدل جدید با وزن و گزارش واقعی منتشرشدنی آماده شود.
- **یافته‌ها / اولویت / اندازه:** F08، F16، F24؛ High؛ Large.
- **ایجاد:** `configs/experiments/baseline_l1.yaml`، `configs/experiments/distribution_variant.yaml`، `experiments/README.md`؛ خروجی‌های کوچک `experiments/<run-id>/config.yaml`, `environment.json`, `metrics.json`, `split_manifest.json` و فایل checksum. اگر train فقط config کامل می‌خواند، هر فایل باید config کامل معتبر باشد یا composition در P1.2 صریح پشتیبانی شود.
- **تغییر:** `Readme.md` برای نتیجهٔ تأییدشده؛ خروجی وزن در Git عادی commit نشود.
- **جزئیات:** یک CNN اقتصادی قبل از sweep پنج backbone؛ split و budget یکسان، validation برای انتخاب checkpoint، test برای گزارش نهایی. MAE/CS و group counts؛ latency با سخت‌افزار، batch، precision، warmup، synchronization و دامنهٔ عملیات. آزمایش‌های چندseed برای ادعای اختلاف کوچک، با هزینهٔ روشن. نتیجهٔ نامطلوب نیز ثبت شود.
- **پیش‌نیاز:** دروازه‌های فاز ۱–۳، داده و compute در دسترس و شرایط روشن استفاده؛ شروع training عملی نیازمند بررسی محیط آن جلسه است.
- **پذیرش:** شخص دیگر با command، environment و manifests بتواند آزمایش را بازاجرا کند؛ prediction CSV با protocol و checkpoint وصل شود؛ جدول هیچ عدد تخمینی به‌جای measurement نداشته باشد.
- **تست:** اجرای واقعی training/evaluation و rerun حداقل inference/eval از artifact در محیط تازه؛ برای reproducibility چند run کوچک در محیط ثابت و ثبت tolerance/محدودیت.
- **خروجی:** اولین release candidate دارای شواهد، نه الزاماً بهترین MAE.

### P4.2 — مستندات منطبق با رفتار و model card

- **هدف:** مسیر نصب تا نتیجه و محدودیت‌ها برای کاربر روشن باشد.
- **یافته‌ها / اولویت / اندازه:** F23، F25؛ High؛ Medium.
- **تغییر:** `Readme.md` و `CONTRIBUTING.md`.
- **ایجاد:** `docs/INSTALLATION.md`، `docs/DATASETS.md`، `docs/REPRODUCIBILITY.md`، `docs/INFERENCE.md`، `docs/MODEL_CARD.md`.
- **جزئیات:** quick start جدا برای وزن آماده و train از صفر، دستور CPU آزموده‌شده، یک checkpoint مشخص به‌جای glob مبهم؛ supported datasets را از related datasets جدا کنید. loss variant، legacy protocol، محدودیت سن 1–84 نمونه‌های فعلی، crop، domain shift، bias و confidence مستند شوند. مدل‌کارت از metrics واقعی P4.1 استفاده کند؛ هر field بدون شاهد، unknown یا not evaluated باشد. کد MIT و شرایط داده/backbone/وزن جدا شوند. معماری Mermaid موجود اصلاح و workflow data→train→eval→infer نمایش داده شود؛ ساخت تصویر تزئینی لازم نیست.
- **پیش‌نیاز:** P2.2/P2.3 و P4.1 برای نتایج؛ حذف ادعای نادرست باید پیش‌تر انجام شده باشد.
- **پذیرش:** هر فرمان با فایل/flag موجود منطبق؛ نمونه‌خروجی measured یا illustrative برچسب داشته باشد؛ عدد 4.73 به‌عنوان validation legacy معرفی شود مگر test واقعاً اندازه‌گیری شده باشد.
- **تست:** walkthrough دستی در محیط مرجع، بررسی لینک داخلی/فایل، اجرای CLIها و تطبیق تعداد split/مقادیر گزارش با artifact. مستندات متن خالص unit test نمی‌خواهند.
- **خروجی:** README کوتاه‌تر و قابل اعتماد با جزئیات در docs.

### P4.3 — نظم artifacts و مثال‌های کم‌هزینه

- **هدف:** نمونهٔ آموزشی و provenance بدون سنگین‌ترکردن clone فراهم شود.
- **یافته‌ها / اولویت / اندازه:** F26، O01؛ Low؛ Small/Medium.
- **ایجاد:** `docs/ARTIFACTS.md`؛ در صورت نیاز `notebooks/01_EDA.ipynb` با ورودی CSV مستند و خروجی کوچک.
- **تغییر:** `Readme.md`، `.gitignore` و شرح نقش `H-parameters.csv`؛ انتقال/حذف legacy فقط پس از تعیین جایگزین و لینک دانلود معتبر.
- **جزئیات:** inventory وزن و event قدیمی با hash/provenance؛ وزن‌های جدید به Release یا مدل‌رجیستری، دادهٔ خام خارج Git. notebook تحلیل توزیع‌ها از CSV، بدون دانلود پنهان یا وابستگی به مسیر شخصی. برای PDFها منبع و شرایط بازتوزیع بررسی و در صورت نامشخص‌بودن، پیشنهاد جایگزینی با لینک رسمی ثبت شود.
- **پیش‌نیاز:** P4.2؛ O01 اختیاری است و مانع انتشار پایه نیست.
- **پذیرش:** artifactهای مورد اشاره قابل پیدا‌کردن، license/source روشن یا unknown صریح؛ هیچ Git history rewrite یا حذف فایل کاربر در این گام ضروری نیست.
- **تست:** checksum artifactهای منتخب، اجرای notebook از ابتدا اگر ساخته شد، بررسی فایل‌های جدید tracked برای نبود secret/وزن/داده خام.
- **خروجی:** منابع آموزشی قابل استفاده و میراث پروژه با شرح روشن.

## فاز ۵ — انتشار و رشد متن‌باز مبتنی بر ارزش

### P5.1 — انتشار وزن و نسخهٔ قابل استفاده

- **هدف:** کاربر بدون آموزش طولانی به نتیجهٔ واقعی برسد.
- **یافته‌ها / اولویت / اندازه:** F03، F25، O07؛ High پیش از انتشار عمومی؛ Medium.
- **ایجاد:** `CHANGELOG.md` و release checklist در `docs/RELEASING.md`؛ workflow انتشار فقط در صورت نیاز نگهداشت‌کننده.
- **تغییر:** لینک‌های artifact و citation در `Readme.md`؛ فایل model card مرتبط با همان وزن.
- **جزئیات:** release tag مرتبط با commit و schema version، وزن معتبر، SHA256، config، metrics، split/environment manifest و شرایط استفادهٔ روشن؛ بدون ادعای تعمیم حقوق MIT به داده. بستهٔ نصب‌شدنی را ابتدا با install smoke تأیید کنید؛ انتشار PyPI برای اولین release الزام ندارد.
- **پیش‌نیاز:** P4.1/P4.2 و بررسی شرایط انتشار artifact.
- **پذیرش:** دریافت وزن طبق checksum، load و inference/eval مطابق مستندات؛ همهٔ unknownها ثبت شده‌اند؛ latest و legacy در نام/لینک اشتباه نشوند.
- **تست:** download/reload در محیط تمیز و تکرار command quick start؛ release فایل credential یا دادهٔ خام نداشته باشد.
- **خروجی:** نسخه‌ای که جامعه می‌تواند فوراً استفاده و ارزیابی کند.

### P5.2 — مسیر کم‌اصطکاک جامعه و demo مشروط

- **هدف:** بازخورد و مشارکت از استفادهٔ واقعی حاصل شود.
- **یافته‌ها / اولویت / اندازه:** F27، O07؛ Medium؛ Small/Medium.
- **ایجاد در صورت انتخاب demo:** `demo/README.md` و فقط فایل deployment لازم برای میزبان منتخب؛ assets نمونهٔ دارای حق استفاده.
- **تغییر:** README برای use case روشن و لینک نتیجه/دمو؛ GitHub About/topics از تنظیمات مخزن توسط maintainer تنظیم شوند.
- **جزئیات:** issueهای محدود با acceptance و good-first-issue واقعی، زمان پاسخ واقع‌بینانه، changelog و راهنمای گزارش failure. demo public فقط با مدل معتبر، crop policy، محدودیت سن و توضیح پردازش/نگهداری عکس؛ token/credential داخل repo قرار نگیرد. برای growth، benchmark و آموزش مفید را در محل‌های مرتبط معرفی کنید؛ ستاره نتیجهٔ تضمین‌شده نیست.
- **پیش‌نیاز:** P3.3 و P5.1؛ public hosting همچنین وابسته به بررسی شرایط F25 است.
- **پذیرش:** کاربر جدید مسیر infer/train/contribute را پیدا کند؛ demo پس از restart وزن ثابت را load کند و ورودی خالی/بدون چهره مطابق policy پاسخ دهد.
- **تست:** walkthrough کاربر تازه، startup smoke میزبان، بررسی privacy behavior و مدل‌کارت؛ هیچ ارسال پیام یا تبلیغ خودکار جزو این گام نیست.
- **خروجی:** discoverability و contribution آسان‌تر، بدون وعدهٔ تعداد star.

### P5.3 — انتخاب یک قابلیت پیشرفته با نیاز واقعی

- **هدف:** فقط قابلیت دارای مصرف‌کننده و تست نگهداشت‌پذیر افزوده شود.
- **اولویت / اندازه:** Low تا Medium؛ بسته به انتخاب.
- **گزینه‌ها و فایل‌های پیشنهادی:**
  - O02: `scripts/inference_onnx.py` و `tests/test_inference_onnx.py` پس از P2.4؛ acceptance برابر بودن preprocessing/age decode و اندازه‌گیری CPU.
  - O03: `scripts/webcam.py` و در صورت نیاز `src/inference/face_detection.py`؛ test صفر/یک/چند چهره، release دوربین و fps واقعی. فقط با نیاز عکس کامل/وبکم.
  - O04: `src/training/ordinal_losses.py` و تست‌های تحلیلی آن؛ مقایسهٔ CORAL با budget و split یکسان. اضافه‌کردن به README پس از نتیجه.
  - O05: `scripts/explain.py` و توضیح محدودیت heatmap؛ sanity check روی نمونه‌ها و بدون ادعای علیت.
  - O06: `src/data/adapters.py` و config دیتاست منتخب؛ پس از دسترسی/مجوز و شناخت identity protocol، تست mapping و عدم نشت هویتی.
  - O08: batching/compile/quantization در مسیرهای موجود؛ تنها پس از profiling، با گزارش before/after و parity؛ fork کردن pipeline برای هر optimization توصیه نمی‌شود.
- **پیش‌نیاز:** baseline منتشرشدهٔ P5.1 و نیاز ثبت‌شده در issue؛ cross-dataset به داده و حقوق مستقل وابسته است.
- **پذیرش:** یک قابلیت انتخاب شود، دامنه و معیار خروج مشخص باشد و کیفیت baseline کاهش نیابد؛ اجراکردن همهٔ گزینه‌ها هدف این نقشه نیست.
- **خروجی:** ارزش پژوهشی یا کاربردی مشخص با هزینهٔ نگهداشت قابل قبول.

## پوشش یافته‌ها و معیار تکمیل

| یافته | گام اصلی |
|---|---|
| F01 / F15 | P1.1، P3.2 |
| F02 / F03 | P1.3، P5.1 |
| F04 / F08 / F09 | P1.4، P4.1 |
| F05 | P1.2 |
| F06 / F07 | P1.5 |
| F10 | P1.2، P2.1 |
| F11 / F12 / F20 | P2.1 |
| F13 / F16 | P2.2 |
| F14 | P1.5، P2.1 |
| F17 | P2.3 |
| F18 | P2.4 |
| F19 | P3.1، P3.2؛ تست رگرسیون در هر گام |
| F21 / F22 | P1.6؛ safe artifact در P1.3 |
| F23 | اصلاح فوری ادعاها، سپس P4.2 |
| F24 | P4.1 |
| F25 | P4.2، P5.1 |
| F26 | P4.3 |
| F27 | P3.3، P5.2 |
| F28 | P1.2، P1.5، P2.2 |
| O01–O08 | P4.3 یا P5.2/P5.3؛ اختیاری |

برای بستن هر گام، PR باید مشکل و رفتار جدید، تست‌های واقعاً اجراشده و نتیجه، محدودیت‌های باقی‌مانده و artifact/تغییر قرارداد را ذکر کند. اگر تست اجرا نشد، علت صریح ثبت شود؛ «تست نوشته شد» به معنی «تست سبز است» نیست. هیچ MAE بهتر، latency کمتر یا star بیشتری معیار تضمین‌شدهٔ تکمیل مهندسی نیست.

**گام اجرایی بعدی: P1.4**؛ پیش از ویرایش، وضعیت working tree بررسی و تغییرات کاربر حفظ شود.
P1.1 تا P1.3 را صرفاً به‌علت وجود شرح برنامه در این سند دوباره از ابتدا اجرا نکنید.
