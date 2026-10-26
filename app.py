import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Adalet Denetim Aracı", layout="wide")
st.title("Adalet Denetim Aracı")
st.caption("Karar verilerinde gruplar arasındaki farkı denetler. "
           "Uygulama yüklenen dosyayı diske yazmaz; veri yalnızca oturum boyunca bellekte tutulur.")

# --- Veri yükleme ---
yuklenen = st.file_uploader("CSV dosyası yükle", type="csv")
if yuklenen is None:
    st.info("Başlamak için bir CSV dosyası yükle.")
    st.stop()

def csv_oku(dosya):
    for kodlama in ("utf-8", "cp1254", "latin-1"):
        for ayirac in (",", ";"):
            dosya.seek(0)
            try:
                d = pd.read_csv(dosya, sep=ayirac, encoding=kodlama)
            except UnicodeDecodeError:
                break
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if d.shape[1] > 1:
                return d
    return None


df = csv_oku(yuklenen)
if df is None or len(df) == 0:
    st.error("Dosya okunamadı: boş olabilir, en az 2 sütunu ve 1 satırı olmalı (ayraç virgül ya da noktalı virgül).")
    st.stop()
st.write(f"{df.shape[0]} satır, {df.shape[1]} sütun")
st.dataframe(df.head())

# --- 1. Filtre (isteğe bağlı) ---
st.subheader("1. Satırları filtrele (isteğe bağlı)")
filtre_sutun = st.selectbox("Filtre sütunu (örn. şirket)", ["(filtre yok)"] + list(df.columns))
if filtre_sutun != "(filtre yok)":
    filtre_degerler = st.multiselect("Hangi değerler kalsın?", list(df[filtre_sutun].dropna().unique()))
    if filtre_degerler:
        df = df[df[filtre_sutun].isin(filtre_degerler)]
        st.write(f"Filtreden sonra {len(df)} satır kaldı.")
    else:
        st.info("Filtre için en az bir değer seç, yoksa tüm satırlar kullanılır.")

# --- 2. Sütun seçimi ---
st.subheader("2. Sütunları seç")
hedef = st.selectbox("Karar sütunu (örn. işe alındı mı?)", list(df.columns))
grup = st.selectbox("Karşılaştırılacak grup sütunu (örn. cinsiyet)",
                    [c for c in df.columns if c != hedef])

degerler = list(df[hedef].dropna().unique())
if len(degerler) != 2:
    st.error(f"Karar sütununda tam 2 farklı değer olmalı, {len(degerler)} tane var: {degerler[:10]}")
    st.stop()
olumlu = st.selectbox("Hangi değer 'olumlu karar' demek?", degerler)

gruplar = list(df[grup].dropna().unique())
if len(gruplar) < 2:
    st.error("Grup sütununda en az 2 farklı değer olmalı.")
    st.stop()
secilen = st.multiselect("Karşılaştırılacak iki grup", gruplar,
                         default=gruplar[:2], max_selections=2)
if len(secilen) != 2:
    st.warning("Tam iki grup seç.")
    st.stop()

veri = df[df[grup].isin(secilen)].dropna(subset=[hedef, grup]).copy()
veri["_olumlu"] = (veri[hedef] == olumlu).astype(int)

# --- 3. Oranlar ---
st.subheader("3. Gruplara göre olumlu karar oranı")
tablo = veri.groupby(grup)["_olumlu"].agg(kişi="size", oran="mean").round(3)
st.dataframe(tablo, width=450)

# --- 4. Adalet oranı ve %80 kuralı ---
st.subheader("4. Adalet oranı (%80 kuralı)")
n = veri.groupby(grup)["_olumlu"].size()
p = veri.groupby(grup)["_olumlu"].mean()
g1, g2 = list(p.index)

if p.max() == 0:
    st.warning("İki grupta da olumlu karar yok, oran hesaplanamaz.")
    st.stop()
if n.min() < 30:
    st.warning(f"Gruplardan biri çok küçük ({int(n.min())} kişi). Sonuçlar güvenilir olmayabilir.")

di = p.min() / p.max()
avantajli = p.idxmax()

# Bootstrap: her grubun olumlu oranını, aynı büyüklükte yeniden örnekleyerek 5000 kez hesapla
rng = np.random.default_rng(0)
B = 5000
b1 = rng.binomial(n[g1], p[g1], B) / n[g1]
b2 = rng.binomial(n[g2], p[g2], B) / n[g2]
yuksek, dusuk = np.maximum(b1, b2), np.minimum(b1, b2)
bs = np.divide(dusuk, yuksek, out=np.full(B, np.nan), where=yuksek > 0)
bs = bs[~np.isnan(bs)]
alt, ust = np.percentile(bs, [2.5, 97.5])

c1, c2, c3 = st.columns(3)
c1.metric("Adalet oranı (DI)", f"{di:.3f}")
c2.metric("%95 aralık", f"{alt:.3f} – {ust:.3f}")
c3.metric("Daha çok olumlu karar alan grup", str(avantajli))

if ust < 0.8:
    st.error(f"Aralığın tamamı 0.8'in altında: %80 kuralı ihlal ediliyor ({avantajli} grubu lehine).")
elif alt >= 0.8:
    st.success("Aralığın tamamı 0.8'in üstünde: %80 kuralı ihlali tespit edilmedi.")
else:
    st.warning("Aralık 0.8'i kapsıyor: ihlal kanıtlanamadı, ama adil olduğu da kanıtlanamadı.")
st.caption("%80 kuralı bir temel ölçüttür, istatistiksel test değildir. "
           "Aralık, grupların yeniden örneklenmesiyle (bootstrap) hesaplanır.")


# --- 5. Doğrudan etki (nitelikler sabit tutulunca) ---
@st.cache_data(show_spinner="Doğrudan etki hesaplanıyor (yaklaşık 10-20 sn)...")
def dogrudan_etki(X, grup_ind, y, B=500):
    Xs = StandardScaler().fit_transform(X)
    Z = np.column_stack([Xs, grup_ind])

    def hesapla(Zb, yb):
        m = LogisticRegression(C=1e6, max_iter=5000).fit(Zb, yb)
        return float(np.exp(m.coef_[0][-1]))

    nokta = hesapla(Z, y)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        bs.append(hesapla(Z[idx], y[idx]))
    alt, ust = np.percentile(bs, [2.5, 97.5])
    return nokta, float(alt), float(ust)


st.subheader("5. Doğrudan etki (nitelikler sabit tutulunca)")
id_benzeri = [c for c in df.columns
              if c not in (hedef, grup)
              and df[c].nunique() > 0.9 * len(df)
              and not pd.api.types.is_float_dtype(df[c])]
adaylar = [c for c in df.columns if c not in (hedef, grup) and c not in id_benzeri]
if id_benzeri:
    st.caption("Kimlik (ID) gibi göründüğü için listeden çıkarılan sütunlar: " + ", ".join(id_benzeri))
ozellikler = st.multiselect(
    "Nitelik sütunları (kararda kullanılan bilgiler; kimlik/ID sütunu seçme)", adaylar)
if not ozellikler:
    st.info("Doğrudan etkiyi hesaplamak için en az bir nitelik sütunu seç.")
    st.stop()
hedef_grup = st.selectbox("Odds oranı hangi grup için hesaplansın?", secilen)

alt_veri = veri.dropna(subset=ozellikler)
atilan = len(veri) - len(alt_veri)
if atilan:
    st.caption(f"Eksik değer içeren {atilan} satır bu analizden çıkarıldı.")
for c in ozellikler:
    if alt_veri[c].nunique() > 0.9 * len(alt_veri):
        st.warning(f"'{c}' sütunu neredeyse her satırda farklı, kimlik (ID) sütunu olabilir. Çıkarmayı düşün.")

X = pd.get_dummies(alt_veri[ozellikler], drop_first=True).astype(float)
grup_ind = (alt_veri[grup] == hedef_grup).astype(float).to_numpy()
y = alt_veri["_olumlu"].to_numpy()

nokta, alt5, ust5 = dogrudan_etki(X.to_numpy(), grup_ind, y)
diger = [g for g in secilen if g != hedef_grup][0]
if not (alt5 <= nokta <= ust5):
    st.warning("Tahmin, kendi güven aralığının dışında kalıyor: model bu sütun seçimiyle kararsız. "
               "Sonucu yorumlama. Sütun seçimini gözden geçir (örn. cinsiyetle çok güçlü bağlantılı "
               "sütunları çıkarmayı dene).")
    st.stop()
d1, d2 = st.columns(2)
d1.metric(f"Odds oranı ({hedef_grup} / {diger})", f"{nokta:.2f}")
d2.metric("%95 aralık", f"{alt5:.2f} – {ust5:.2f}")

if alt5 > 1:
    st.error(f"Nitelikler sabitken {hedef_grup} grubunun olumlu karar odds'u {diger} grubunkinden "
             f"yaklaşık {nokta:.2f} kat yüksek. Doğrudan etki tespit edildi.")
elif ust5 < 1:
    st.error(f"Nitelikler sabitken {hedef_grup} grubunun olumlu karar odds'u {diger} grubunkinden "
             f"yaklaşık {1 / nokta:.2f} kat düşük. Doğrudan etki tespit edildi.")
else:
    st.warning("Doğrudan etki tespit edilemedi (aralık 1'i kapsıyor). Bu, etki olmadığı anlamına "
               "gelmez; örneklem küçükse bu analizin gücü düşüktür.")
st.caption("Odds oranı 1 ise fark yoktur. Yalnızca seçtiğin nitelikler sabit tutulur; "
           "seçmediğin nitelikler fark yaratıyor olabilir. Aralık, satırların yeniden örneklenmesiyle (bootstrap) hesaplanır.")



# --- 6. Model denetimi ve düzeltme ---
@st.cache_data(show_spinner="Model ve düzeltme hesaplanıyor...")
def model_ve_duzeltme(X, grup_ind, y, eg=False, n_bolme=30):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from fairlearn.metrics import demographic_parity_ratio, equalized_odds_difference
    from fairlearn.postprocessing import ThresholdOptimizer

    grup_ind = grup_ind.astype(int)
    strat = grup_ind * 2 + y
    kayit = []

    def ekle(ad, yte, tahmin, gte):
        tahmin = np.asarray(tahmin)
        kayit.append({
            "yöntem": ad,
            "doğruluk": accuracy_score(yte, tahmin),
            "DI": demographic_parity_ratio(yte, tahmin, sensitive_features=gte),
            "EO_farkı": equalized_odds_difference(yte, tahmin, sensitive_features=gte),
        })

    for seed in range(n_bolme):
        Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
            X, y, grup_ind, test_size=0.3, random_state=seed, stratify=strat)
        m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        ekle("Düzeltme yok", yte, m.predict(Xte), gte)
        for kisit in ["demographic_parity", "equalized_odds"]:
            try:
                to = ThresholdOptimizer(estimator=m, constraints=kisit, objective="accuracy_score",
                                        prefit=True, predict_method="predict_proba")
                to.fit(Xtr, ytr, sensitive_features=gtr)
                ekle("ThresholdOptimizer · " + kisit, yte,
                     to.predict(Xte, sensitive_features=gte, random_state=seed), gte)
            except Exception:
                continue
        if eg:
            from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
            for kisit, ad in [(DemographicParity(), "ExponentiatedGradient · demographic_parity"),
                              (EqualizedOdds(), "ExponentiatedGradient · equalized_odds")]:
                try:
                    e = ExponentiatedGradient(LogisticRegression(max_iter=1000), constraints=kisit)
                    e.fit(Xtr, ytr, sensitive_features=gtr)
                    ekle(ad, yte, e.predict(Xte, random_state=seed), gte)
                except Exception:
                    continue

    sonuc = pd.DataFrame(kayit).groupby("yöntem").mean().round(3)
    sira = ["Düzeltme yok"] + [s for s in sonuc.index if s != "Düzeltme yok"]
    return sonuc.loc[sira]


st.subheader("6. Model ve düzeltme")
st.caption("Grup bilgisi modele girdi olarak verilmez (cinsiyetsiz model). 30 farklı rastgele bölmenin "
           "ortalaması gösterilir. Seçtiğin nitelikler (bölüm 5) model girdisidir.")
eg_ac = st.checkbox("ExponentiatedGradient'ı da çalıştır (yavaş, 1-2 dk)")
sayilar = pd.Series(grup_ind.astype(int) * 2 + y).value_counts()
model_uygun = len(sayilar) == 4 and sayilar.min() >= 10
if not model_uygun:
    st.warning("Model için her grup ve karar birleşiminde (grup × olumlu/olumsuz) en az 10 kişi gerekir. "
               "Bu veride sağlanmıyor, bu bölüm çalıştırılamaz.")
sonuc = None
if st.checkbox("Modeli ve düzeltmeyi çalıştır") and model_uygun:
    sonuc = model_ve_duzeltme(X.to_numpy(), grup_ind, y, eg=eg_ac)
    st.dataframe(sonuc)
    st.write(f"Karşılaştırma için: etiketlerdeki DI = {di:.3f}, "
             f"herkese olumsuz karar verilseydi doğruluk = {1 - y.mean():.3f}.")
    st.caption("DI: 1'e yakın daha adil. EO_farkı: 0'a yakın daha adil (hata oranları gruplar arasında benzer). "
               "ThresholdOptimizer karar anında grup bilgisini kullanır, bu hukuken tartışmalıdır; "
               "burada ulaşılabilir adaleti ve doğruluk bedelini göstermek için kullanılır. "
               "Doğruluk, kararların kendisine (geçmiş kararlara) uyum olarak ölçülür; "
               "geçmiş kararlar önyargılıysa 'doğruluk kaybı' gerçek bir kayıp olmayabilir.")

    

# --- 7. Rapor ---
st.subheader("7. Rapor")


def rapor_yaz():
    s = []
    s.append("# Adalet Denetim Raporu\n")
    s.append(f"**Veri:** {len(veri)} kişi, gruplar: {g1} ({int(n[g1])}) ve {g2} ({int(n[g2])}). "
             f"Karar sütunu: {hedef}, olumlu karar: {olumlu}.\n")
    s.append("## 1. Olumlu karar oranları\n")
    for g in p.index:
        s.append(f"- {g}: %{p[g] * 100:.1f}")
    s.append(f"\nAdalet oranı (DI) = {di:.3f}, %95 aralık {alt:.3f} – {ust:.3f}. ")
    if ust < 0.8:
        s.append(f"Aralığın tamamı 0.8'in altında: %80 kuralı ihlal ediliyor ({avantajli} lehine).\n")
    elif alt >= 0.8:
        s.append("Aralığın tamamı 0.8'in üstünde: ihlal tespit edilmedi.\n")
    else:
        s.append("Aralık 0.8'i kapsıyor: ihlal kanıtlanamadı, adil olduğu da kanıtlanamadı.\n")
    s.append("## 2. Doğrudan etki\n")
    s.append(f"Sabit tutulan nitelikler: {', '.join(ozellikler)}. "
             f"Odds oranı ({hedef_grup} / {diger}) = {nokta:.2f}, %95 aralık {alt5:.2f} – {ust5:.2f}. ")
    if alt5 > 1 or ust5 < 1:
        s.append("Aralık 1'i kapsamıyor: nitelikler sabitken bile gruplar arasında fark var.\n")
    else:
        s.append("Aralık 1'i kapsıyor: doğrudan etki tespit edilemedi (bu, etki olmadığını göstermez).\n")
    if sonuc is not None:
        s.append("## 3. Model ve düzeltme (30 bölme ortalaması)\n")
        s.append("| yöntem | doğruluk | DI | EO_farkı |\n|---|---|---|---|")
        for ad, r in sonuc.iterrows():
            s.append(f"| {ad} | {r['doğruluk']:.3f} | {r['DI']:.3f} | {r['EO_farkı']:.3f} |")
        s.append(f"\nKarşılaştırma: etiketlerdeki DI = {di:.3f}; herkese olumsuz karar verilseydi doğruluk = {1 - y.mean():.3f}.\n")
    else:
        s.append("## 3. Model ve düzeltme\n\nBu bölüm çalıştırılmadı.\n")
    s.append("## Sınırlar\n")
    s.append("- Doğrudan etki yalnızca seçilen niteliklere göre hesaplanır; seçilmeyen nitelikler farkı açıklıyor olabilir.")
    s.append("- Doğruluk, geçmiş kararlara uyumdur; geçmiş kararlar önyargılıysa doğruluk kaybı gerçek bir kayıp olmayabilir.")
    s.append("- Bu araç istatistiksel bir kanıt sunar, hukuki bir ayrımcılık kararı vermez.")
    s.append("- Veri geçmiş kararlardır; canlı bir sistem değildir.")
    s.append("- ThresholdOptimizer karar anında grup bilgisini kullanır ve bu hukuken tartışmalıdır; ExponentiatedGradient yalnızca eğitimde kullanır.")
    s.append("- Eşit seçim oranı (DI) ile eşit hata oranları (EO_farkı) aynı anda sağlanamayabilir; bir yöntem birini iyileştirirken diğerini kötüleştirebilir.")
    return "\n".join(s)


rapor = rapor_yaz()
st.markdown(rapor)
st.download_button("Raporu indir (.md)", rapor, file_name="adalet_raporu.md")



# --- 8. Claude ile açıklama ---
import os
st.subheader("8. Claude ile yorum (isteğe bağlı)")
st.caption("Yukarıdaki rapor, hiçbir sayı eklenmeden akıcı bir metne çevrilir. "
           "Rapor metni (kişisel veri değil, yalnızca özet sayılar) Anthropic API'ye gönderilir.")
anahtar = os.environ.get("ANTHROPIC_API_KEY")
if not anahtar:
    try:
        anahtar = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        anahtar = None
if not anahtar:
    st.info("Bu özellik için API anahtarı tanımlı olmalı (.streamlit/secrets.toml). Tanımlı değil, bölüm atlandı.")
elif st.button("Claude ile yorum yaz"):
    try:
        import anthropic
        istemci = anthropic.Anthropic(api_key=anahtar)
        yanit = istemci.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1200,
                        system=("Sen bir adalet denetimi raporunu sade Türkçeyle özetleyen yardımcısın. "
                    "Yalnızca verilen rapordaki sayıları ve ifadeleri kullan; yeni sayı, tahmin veya iddia ekleme. "
                    "Nedensellik iddia etme: 'yüzünden', 'sebep oluyor' gibi ifadeler kullanma. "
                    "Bir sayıyı verirken hangi yöntemden geldiğini söyle. "
                    "Tablodaki düzeltme yöntemlerini tek tek an ve her birinin hem kazancını hem bedelini "
                    "(doğruluk, DI, EO_farkı) aynı cümlede ver; yalnızca olumlu sonucu öne çıkarma. "
                                        "Sütun adlarını çevirme ya da yorumlama, raporda yazıldığı gibi kullan. "
                    "İyileşmeyi 'orta', 'yüksek' gibi sıfatlarla niteleme; sayıyı ver ve 0.8 eşiğiyle karşılaştır. "
                    "Verinin geçmiş kararlar olduğunu, canlı bir sistem olmadığını unutma. "
                    "Raporun 'Sınırlar' bölümünü olduğu gibi yansıt, yumuşatma ya da güçlendirme. "
                    "3 kısa paragraf yaz: ne bulundu, ne anlama geliyor (yalnızca rapordakilerle), nelere dikkat edilmeli."),
            messages=[{"role": "user", "content": rapor}],
        )
        yorum = yanit.content[0].text
        st.markdown(yorum)
        st.download_button("Yorumu indir (.md)", yorum, file_name="adalet_yorumu.md")
    except Exception as e:
        st.error(f"Claude API çağrısı başarısız: {e}")