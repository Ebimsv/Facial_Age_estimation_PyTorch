# بهبود ارزش متن‌باز، تجربهٔ کاربر و جامعه

تاریخ: ۲۰۲۶-۱۱-۰۶  
مرجع فنی: [ممیزی مخزن](REPOSITORY_AUDIT.md)  
ترتیب اجرا: [نقشهٔ راه](IMPLEMENTATION_ROADMAP.md)

## ۱. پیشنهاد اصلی

بیشترین ارزش فعلی از یک مسیر کوتاه و قابل اعتماد برای «نصب → دریافت وزن معتبر → inference → ارزیابی قابل‌بازتولید» حاصل می‌شود. پروژه از قبل README تصویری، نمودار معماری، چند backbone، دمو، راهنمای مشارکت و CI دارد؛ ایجاد دوبارهٔ این موارد سود اصلی نیست. لازم است این امکانات به checkpoint معتبر و تست مسیر اصلی متصل شوند و ادعاها دقیق باشند.

رشد star و contribution قابل تضمین نیست. شاخص اصلی موفقیت باید این باشد که شخص دیگر بتواند پروژه را اجرا، نتیجه را بررسی و مشکل مشخصی را اصلاح کند. هیچ خرید/مبادلهٔ star، تبلیغ انبوه، benchmark ساختگی یا افزایش بی‌هدف قابلیت‌ها پیشنهاد نمی‌شود.

## ۲. آنچه باید حفظ شود و آنچه باید اصلاح شود

| دارایی فعلی | وضعیت | اقدام پیشنهادی | اولویت / اندازه |
|---|---|---|---|
| banner، تصاویر و فهرست README | Complete در ارائهٔ بصری | حفظ بخش‌های مفید؛ کوتاه‌کردن تکرار و انتقال جزئیات به docs | Low / Small |
| نمودار Mermaid معماری | Partial از نظر دقت همهٔ modeها | افزودن regression branch، support قابل تنظیم و تفکیک train/infer؛ CNNها و MobileNet جزئیات یکسان ندارند | Medium / Small |
| پنج backbone | Complete در کد، اجرا در ممیزی تأیید نشده | معرفی به‌عنوان implemented؛ tested/benchmarked را جدا مشخص کنید؛ افزودن مدل جدید فعلاً اولویت ندارد | Medium / Small |
| جدول نتایج | Partial | حذف latencyهای بدون روش، مشخص‌کردن legacy validation و حفظ TBD برای آزمایش اجرا‌نشده | High / Small |
| Quick Start | Partial | دستورهای نصب و وزن مشخص با smoke test؛ دو مسیر inference آماده و training | High / Medium |
| Gradio | Partial | قرارداد artifact مشترک، خطای ورودی، مدل‌کارت و سپس میزبانی مشروط | Medium / Medium |
| CONTRIBUTING و CI | Partial | هماهنگی با tooling واقعی و تست نصب/pipeline؛ templateهای کوتاه | High / Medium |
| منابع علمی | Partial | پیوند رسمی و تفاوت implementation با روش مرجع؛ بررسی حقوق assets | Medium / Small |

هیچ عنوان «production-ready»، رتبهٔ SOTA، بهبود عددی ثابت loss یا تضمین زمان آموزش نباید بدون شواهد متناسب استفاده شود. جایگزین روشن اولیه می‌تواند «خط لولهٔ پژوهشی PyTorch برای تخمین سن از چهرهٔ crop‌شده، همراه آموزش، ارزیابی و inference» باشد؛ وضعیت verification در همان نزدیکی توضیح داده شود.

## ۳. ساختار پیشنهادی README

README برای انتخاب و شروع سریع باشد؛ جزئیات تکرارپذیری در اسناد تخصصی بماند.

1. **مسئله و دامنه:** تخمین سن از single cropped face؛ دادهٔ اصلی UTKFace؛ خروجی سن تخمینی و حالت توزیعی؛ محدودیت سنی مدل آموزش‌دیده.
2. **یک نمونهٔ قابل تکرار:** تصویر دارای حق استفاده، نام checkpoint، command و prediction واقعی همان artifact. نمونهٔ تصویر فعلی بدون provenance به‌عنوان benchmark استفاده نشود.
3. **شروع سریع inference:** محیط مرجع، نصب خود بسته، دریافت artifact مشخص و checksum، یک فرمان با مسیر دقیق وزن. glob مانند `best_*.pt` در راه اصلی نباشد؛ صفر یا چند match مبهم است.
4. **شروع سریع training:** دریافت مجاز cropped dataset، ساخت protocol، اجرای train، resume و evaluate با config مشخص؛ تفاوت هزینهٔ inference و آموزش روشن شود.
5. **نتایج واقعی و protocol:** جدول فقط measurement، نام split و artifact؛ سایر مدل‌ها در جدول قابلیت‌ها یا «ارزیابی‌نشده» قرار گیرند.
6. **نمودار workflow و معماری:** استفاده از Mermaid موجود؛ فقط رفتار واقعی را نمایش دهد. ورود عکس کامل بدون detection نباید ترسیم شود.
7. **محدودیت‌ها و شرایط:** label noise، domain shift، گروه‌های جمعیتی، confidence کالیبره‌نشده و تفکیک شرایط کد/داده/وزن.
8. **لینک مستندات و مشارکت:** train/inference/export، model card، گزارش bug و گام‌های کوچک roadmap.
9. **citation و license:** BibTeX فعلی قابل حفظ است؛ `CITATION.cff` زمانی که اطلاعات نسخه/ارجاع آماده شد اختیاری است.

نشان‌های Python، PyTorch و CI باید به نسخه‌ها و مسیرهای واقعاً آزموده‌شده مربوط باشند. badge تعداد تست یا درصد پوشش جای شواهد صحت را نمی‌گیرد. توضیحات علمی طولانی و فهرست دیتاست‌های مرتبط به `docs/` منتقل شوند؛ عنوان «Supported Datasets» فقط برای adapterهای واقعاً پشتیبانی‌شده استفاده شود.

## ۴. بستهٔ آزمایش قابل‌بازتولید

برای هر نتیجهٔ منتشرشده، حداقل این اقلام کوچک در دسترس باشد:

| مورد | محتوای لازم | علت |
|---|---|---|
| شناسهٔ آزمایش | run-id، Git commit و وضعیت dirty tree | اتصال نتیجه به کد واقعی |
| config نهایی | مقادیر resolved، loss variant و seed | command به‌تنهایی همهٔ تنظیمات را نشان نمی‌دهد |
| protocol داده | source/version، فیلتر سن، seed/روش split و hash CSVها | جلوگیری از مقایسهٔ splitهای متفاوت |
| محیط | Python، torch/torchvision/timm و نسخهٔ CUDA در صورت استفاده | بررسی compatibility و تکرار آزمایش |
| artifact مدل | schema version، preprocessing، support سن و SHA256 | بارگذاری درست و بررسی یکپارچگی |
| نتیجه | validation انتخاب مدل جدا از test، n، MAE و CS@5/CS@10 | جلوگیری از معرفی validation به‌عنوان test |
| گروه‌ها | تعداد، معیار و تعریف گروه/age-bin؛ محدودیت گروه کوچک | تفسیر bias بدون پنهان‌کردن n |
| predictions | image id، target، predicted age و error؛ مطابق شرایط داده | بررسی خطا و بازحساب معیار |
| روش زمان‌گیری | سخت‌افزار، precision، batch، رزولوشن، warmup و دامنهٔ timing | latency بدون روش قابل مقایسه نیست |

قالب جدول نتیجه، تا قبل از اندازه‌گیری:

| Model / Artifact | Protocol | Input | Loss | Seed(s) | Test n | MAE | CS@5 | Latency / Device |
|---|---|---|---|---|---|---|---|---|
| ارزیابی‌نشده | تعیین‌نشده | از config | variant دقیق | ثبت خواهد شد | اندازه‌گیری‌نشده | — | — | اندازه‌گیری‌نشده |

این ردیف صرفاً قالب است؛ نتیجهٔ مخزن نیست. عدد 4.73 فعلی منشأ validation میراثی دارد. برای مدل‌های جدید و برای latency هیچ عددی از روی مقاله، اندازهٔ معماری یا تصویر قدیمی پر نشود. تعداد پارامترها نیز برای head جدید اندازه‌گیری شود؛ اعداد classifier استاندارد ImageNet لزوماً شمار پارامتر مدل این پروژه نیستند.

برای مقایسهٔ منصفانه، split، فیلتر سن، preprocessing، budget آموزش و سیاست انتخاب checkpoint همسان باشند. آغاز با یک CNN و L1 در برابر variant توزیعی از sweep پنج backbone مفیدتر است. مقالهٔ مرجع منبع توضیح روش است؛ نتیجهٔ مقاله روی protocol دیگر، نتیجهٔ این مخزن محسوب نمی‌شود. اگر اختلاف کوچک است، چند seed و بازهٔ تغییرپذیری پیش از ادعای برتری منتشر شود.

## ۵. وزن آماده، انتشار و مدل‌کارت

### اولویت بالا: artifact قابل استفاده

وضعیت فعلی Partial است: وزن قدیمی موجود است، اما با loader و head جدید سازگار نیست و artifact جدید benchmarkشده نداریم. وابستگی‌ها F02/F03/F24/F25 و گام‌های P1.3، P4.1، P5.1 هستند؛ اندازهٔ کار Medium برای بسته‌بندی artifact و Large برای آموزش/آزمایش.

برای release اول، یک وزن معتبر کفایت می‌کند. هر انتشار شامل tag کد، config، preprocessing، support سن، checksum، protocol و metrics متناظر باشد. نگهداری وزن در GitHub Releases یا مدل‌رجیستری منتخب به‌جای commit فایل‌های جدید سنگین در Git مناسب است؛ انتخاب میزبان به شرایط انتشار و workflow maintainer وابسته است. وزن legacy می‌تواند با نام و راهنمای مستقل حفظ شود؛ تبدیل معماری ناموفق نباید با `strict=False` پنهان شود.

مدل‌کارت پیشنهادی:

- نام و نسخه، backbone و منشأ pretraining، loss دقیق و تنظیمات آموزش؛
- source و شرایط داده، فیلترها و protocol؛
- سن‌های واقعاً دیده‌شده، crop/alignment، normalization و اندازهٔ تصویر؛
- نتایج اندازه‌گیری‌شده و تفکیک validation/test؛
- خطا بر گروه‌ها با تعداد، محدودیت نمونه‌های کم و label noise؛
- کاربردهای داخل و خارج دامنه، محدودیت pose/lighting/occlusion و عکس بدون چهره؛
- توضیح اینکه توزیع softmax، confidence یا احتمال خطای تضمین‌شده نیست؛
- شرایط استفاده و توزیع وزن، کد، نمونه‌تصاویر و backbone به‌صورت جدا؛
- روش بازتولید و issue link برای failureهای قابل تکرار.

شرایط رسمی UTKFace، داده را برای پژوهش غیرتجاری معرفی می‌کند؛ MIT مخزن این شرط را تغییر نمی‌دهد. وضعیت مجازبودن بازتوزیع هر وزن یا میزبانی demo از این جمله به‌تنهایی نتیجه‌گیری نمی‌شود و باید طبق منابع مربوط روشن شود. [صفحهٔ رسمی UTKFace](https://susanqq.github.io/UTKFace/)

### اولویت متوسط و مشروط: دموی عمومی

دموی محلی از قبل موجود است؛ سایت یا سرویس جدید پیش‌نیاز نیست. اگر میزبانی عمومی مناسب باشد، startup باید artifact ثابت را با checksum دریافت کند، UI نام مدل و محدودیت را نشان دهد، ورودی None/خراب را مدیریت کند و کاربر را به cropped face راهنمایی کند. دربارهٔ پردازش و نگهداری عکس‌ها توضیح واقعی داده شود؛ رفتار privacy را از تنظیمات میزبان بررسی کنید و سیاست ساختگی ننویسید.

معیار پذیرش: کاربر بدون setup محلی نمونهٔ دارای حق استفاده را امتحان کند، مدل‌کارت را پیدا کند و برای ورودی خارج دامنه پیام روشن ببیند. برچسب سن تخمینی به‌عنوان اندازه‌گیری قطعی سن یا ابزار احراز هویت معرفی نشود. این توصیه به دامنهٔ خود محصول مربوط است و مستلزم افزودن سامانهٔ پیچیدهٔ compliance نیست.

## ۶. تجربهٔ مشارکت و نگهداشت

CONTRIBUTING موجود است؛ تکمیل آن باید کم‌حجم و عملی باشد:

| اقدام | وضعیت فعلی | نتیجهٔ مورد انتظار | وابستگی | اندازه / اولویت |
|---|---|---|---|---|
| دستور dev روی CPU | Partial؛ CUDA در مثال اصلی | contributor بدون GPU هم suite پایه را اجرا کند | P1.1/P3.2 | Small / High |
| hooks واقعی | Missing؛ دستور install وجود دارد، config نیست | ruff check/format با policy هماهنگ CI | P3.3 | Small / Medium |
| bug report template | Missing | command، environment، traceback و expected/actual؛ بدون secret یا الزام upload چهره | P3.3 | Small / Medium |
| feature request template | Missing | use case، ارزش و دامنه پیش از افزودن dependency | P3.3 | Small / Medium |
| PR template | Missing | شرح رفتار، تست اجراشده، اثر بر artifact/config و docs | P3.3 | Small / Medium |
| release checklist/changelog | Missing | breaking changes و checkpoint migration قابل پیگیری | P5.1 | Small / Medium |
| citation machine-readable | Missing / Optional | ارجاع نسخه‌دار آسان‌تر با CITATION.cff | اطلاعات نسخهٔ معتبر | Small / Low |
| راهنمای پشتیبانی و امنیت | Partial / Optional | محل گزارش bug عمومی و مسیر امن گزارش secret مشخص | ظرفیت واقعی maintainer | Small / Low |

Issueهای مناسب شروع، باید کوچک و دارای شاهد و acceptance باشند؛ مثال‌های قابل تبدیل به issue:

1. افزودن تست تحلیلی MAE/CS و مرز `<=5`؛ معیار خروج: مقادیر محاسبه‌شدهٔ دستی و ورودی خالی پوشش داده شوند.
2. ساخت parent خروجی evaluation و افزودن image id؛ معیار خروج: فرمان README در پوشهٔ تمیز کار کند و هر prediction به تصویر متصل باشد.
3. اصلاح metadata و config resource بسته؛ معیار خروج: wheel خارج checkout نصب و CLI help اجرا شود. این مورد نیازمند تجربهٔ packaging است و لزوماً good-first-issue نیست.
4. برگرداندن مدل/support واقعی در دمو؛ معیار خروج: regression و distribution config هر دو UI درست بدهند.
5. اصلاح docstring transform و early stopping؛ معیار خروج: متن رفتار فعلی را دقیق شرح دهد، بدون افزودن قابلیت ناخواسته.

Issueهای correctness مثل checkpoint schema و resume را به‌عنوان کار سادهٔ تازه‌کار برچسب نزنید. تعداد issue باز شاخص کیفیت جامعه نیست. good-first-issue فقط وقتی نگه‌دارنده قادر به توضیح دامنه و review است مفید خواهد بود.

## ۷. discoverability مبتنی بر کاربرد واقعی

پس از تثبیت quick start و اولین نتیجه:

- GitHub About مسئله، PyTorch و cropped-face age estimation را روشن بیان کند. topicهای مرتبط موجودند؛ افزودن keyword نامرتبط یا نام دیتاستِ بدون adapter توصیه نمی‌شود.
- اولین بخش README یک use case مشخص و یک command قابل اجرا داشته باشد؛ دعوت کوتاه به star اختیاری است و نباید محتوای فنی را تحت‌الشعاع قرار دهد.
- یک tutorial قابل تکرار مانند «train/evaluate یک CNN سبک با protocol مشخص» منتشر شود؛ EDA از CSV و یک گزارش failure analysis ارزشمندتر از گالری benchmark فرضی است.
- بعد از release، معرفی دستی و مرتبط با جامعهٔ PyTorch/بینایی ماشین می‌تواند مفید باشد؛ پیام انبوه و خودکار یا وعدهٔ نتیجه توصیه نمی‌شود. ارسال هیچ پیام بیرونی جزو این ممیزی نیست.
- مقایسه با CORAL یا روش مرجع تنها پس از implementation و protocol همسان انجام شود. برای cross-dataset، محدودیت دسترسی، labels و identity split از قبل مشخص شود.

## ۸. معیارهای سنجش پیشرفت

| معیار مفید | روش سنجش |
|---|---|
| موفقیت شروع سریع | اجرای دستورها توسط محیط تمیز و ترجیحاً یک کاربر تازه؛ ثبت failureها |
| قابل‌بازتولید بودن نتیجه | artifact و protocol کامل، بازاجرای eval و اختلاف عددی مستند |
| قابلیت نصب | wheel smoke در CI، شامل resource و entry pointهای لازم |
| سودمندی demo | startup پایدار، ورودی نامعتبر مدیریت‌شده و مدل‌کارت در دسترس |
| کیفیت مشارکت | issueهای actionable، PRهای کوچک و review روشن؛ بدون وعدهٔ SLA غیرواقعی |
| سلامت نگهداشت | release notes، سازگاری artifact و dependency، عدم تکرار regressionهای شناخته‌شده |
| star/fork | شاخص ثانویهٔ علاقه، بدون هدف تضمینی یا نسبت‌دادن علت بدون شواهد |

پیشنهاد انتشار اولیه، پس از عبور از دروازه‌های فازهای ۱–۴ است: یک مدل، یک protocol، یک نتیجهٔ واقعی، یک مسیر inference معتبر و راه مشارکت روشن. قابلیت‌های پیشرفته فقط وقتی به این پایه ارزش مشخص اضافه می‌کنند وارد roadmap اجرایی شوند.
