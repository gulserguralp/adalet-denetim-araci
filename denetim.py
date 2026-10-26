"""Adalet denetimi fonksiyonları (Utrecht Fairness Recruitment verisi)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (MetricFrame, selection_rate,
                               demographic_parity_ratio, equalized_odds_difference)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds


def veri_hazirla(df, sirket):
    """Bir şirketin female/male adaylarını (X, y, cinsiyet) olarak döndürür."""
    d = df[(df['company'] == sirket) & (df['gender'].isin(['female', 'male']))].copy()
    cols = [c for c in df.columns if c.startswith('ind-')]
    X = pd.get_dummies(d[cols], columns=['ind-degree'], drop_first=True).astype(float)
    return X, d['decision'], d['gender']


def olc(y_true, y_pred, A_t):
    """Doğruluk, DI, EO farkı ve grup seçim oranları."""
    y_pred = np.asarray(y_pred)
    sr = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred,
                     sensitive_features=A_t).by_group
    return {
        'doğruluk': accuracy_score(y_true, y_pred),
        'DI': demographic_parity_ratio(y_true, y_pred, sensitive_features=A_t),
        'EO_farkı': equalized_odds_difference(y_true, y_pred, sensitive_features=A_t),
        'kadın_oran': sr['female'],
        'erkek_oran': sr['male'],
    }


def denetle(df, sirket, seeds=range(30), eg=True):
    """Düzeltme yok + ThresholdOptimizer + ExponentiatedGradient, bölme bölme."""
    X, y, A = veri_hazirla(df, sirket)
    strat = A.astype(str) + "_" + y.astype(str)
    kayit = []
    for seed in seeds:
        Xtr, Xte, ytr, yte, Atr, Ate = train_test_split(
            X, y, A, test_size=0.3, random_state=seed, stratify=strat)
        m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        kayit.append({'şirket': sirket, 'seed': seed, 'yöntem': 'Düzeltme yok',
                      **olc(yte, m.predict(Xte), Ate)})
        for kisit in ['demographic_parity', 'equalized_odds']:
            to = ThresholdOptimizer(estimator=m, constraints=kisit,
                                    objective='accuracy_score', prefit=True,
                                    predict_method='predict_proba')
            to.fit(Xtr, ytr, sensitive_features=Atr)
            p = to.predict(Xte, sensitive_features=Ate, random_state=seed)
            kayit.append({'şirket': sirket, 'seed': seed, 'yöntem': 'TO_' + kisit,
                          **olc(yte, p, Ate)})
        if eg:
            for ad, kisit in [('EG_demographic_parity', DemographicParity()),
                              ('EG_equalized_odds', EqualizedOdds())]:
                e = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=kisit)
                e.fit(Xtr, ytr, sensitive_features=Atr)
                kayit.append({'şirket': sirket, 'seed': seed, 'yöntem': ad,
                              **olc(yte, e.predict(Xte, random_state=seed), Ate)})
    return pd.DataFrame(kayit)


def cevir(df, sirket, seeds=range(30)):
    """Cinsiyet çevirme testi: cinsiyeti gören model, test setinde cinsiyeti ters çevir."""
    X, y, A = veri_hazirla(df, sirket)
    Xg = X.copy()
    Xg['male'] = (A == 'male').astype(float)
    strat = A.astype(str) + "_" + y.astype(str)
    kayit = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            Xg, y, test_size=0.3, random_state=seed, stratify=strat)
        m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        p0 = m.predict_proba(Xte)[:, 1]
        Xc = Xte.copy()
        Xc['male'] = 1 - Xc['male']
        p1 = m.predict_proba(Xc)[:, 1]
        kadin = (Xte['male'] == 0).values
        kayit.append({
            'şirket': sirket, 'seed': seed,
            'kadın→erkek': (p1[kadin] - p0[kadin]).mean(),
            'erkek→kadın': (p1[~kadin] - p0[~kadin]).mean(),
            'OR_erkek/kadın': np.exp(m.coef_[0][list(Xg.columns).index('male')]),
        })
    return pd.DataFrame(kayit)


def X_kur(d, set_adi):
    """Ablasyon için özellik kümeleri."""
    ind = [c for c in d.columns if c.startswith('ind-')]
    kolonlar = {'ind-*': ind,
                'sport': ['sport'],
                'ind-* + sport': ind + ['sport'],
                'ind-* + sport + yaş + uyruk': ind + ['sport', 'age', 'nationality']}[set_adi]
    kat = [c for c in kolonlar if c in ('ind-degree', 'sport', 'nationality')]
    return pd.get_dummies(d[kolonlar], columns=kat, drop_first=True).astype(float)