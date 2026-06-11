# Statistical Hypothesis Analyzer

Bu sadə Streamlit app Excel/CSV datanı analiz edir və avtomatik hipotez nəticələri verir.

## Funksiyalar

- Excel/CSV upload
- Descriptive statistics
- Automatic hypothesis/test suggestions
- Independent Samples t-test
- Paired Samples t-test
- One-way ANOVA
- Tukey post-hoc test
- Pearson/Spearman correlation
- Chi-square test
- Linear regression
- Excel və Word report export

## Quraşdırma

Terminal/CMD aç və bu qovluğa daxil ol:

```bash
cd stat_analysis_app
```

Kitabxanaları yüklə:

```bash
pip install -r requirements.txt
```

App-i işə sal:

```bash
streamlit run app.py
```

Brauzerdə avtomatik açılacaq.

## Qərar qaydası

Default alpha = 0.05.

- p-value < 0.05 → H0 rejected / H1 accepted
- p-value ≥ 0.05 → H0 not rejected / H1 declined

## Qeyd

Statistik qərarlar avtomatik verilir, amma akademik işlərdə assumption check, sample size və data keyfiyyəti ayrıca nəzərə alınmalıdır.