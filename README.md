# Adalet Denetim Aracı

Karar verilerinde (işe alım gibi) iki grup arasındaki farkı denetleyen, herhangi bir CSV ile çalışan Streamlit uygulaması. Üniversite bitirme projesi kapsamında geliştirildi.

## Ne yapar

1. **Olumlu karar oranları** ve adalet oranı (DI = en düşük oran / en yüksek oran), bootstrap ile %95 aralık, %80 kuralı yorumu.
2. **Doğrudan etki:** seçilen nitelikler sabit tutulunca cinsiyet (ya da seçilen grup) için odds oranı ve aralığı (lojistik regresyon + bootstrap).
3. **Model ve düzeltme:** cinsiyetsiz bir model eğitir, Fairlearn'ün ThresholdOptimizer ve ExponentiatedGradient yöntemleriyle düzeltir; doğruluk, DI ve EO_farkını 30 rastgele bölmenin ortalaması olarak gösterir.
4. **Rapor:** sonuçlardan Markdown rapor üretir. İsteğe bağlı olarak Claude API ile sade bir yorum yazdırır.

## Kurulum

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Claude yorumu (isteğe bağlı)

Anahtar olmadan uygulama çalışır, yalnızca bölüm 8 atlanır. Anahtarı kullanmak için `.streamlit/secrets.toml` dosyası oluştur:

```
ANTHROPIC_API_KEY = "sk-ant-..."
```

Bu dosya `.gitignore`'dadır, GitHub'a yüklenmez. Claude'a yalnızca rapordaki özet sayılar gönderilir, satır düzeyinde veri gönderilmez.

## Veri

Geliştirme ve test için Utrecht Fairness Recruitment veri seti kullanıldı (4 şirket × 1000 satır, sentetik). Veri bu depoya dahil değildir. Kendi CSV'ni yükleyebilirsin; karar, grup ve olumlu karar değeri arayüzden seçilir. Grup sütununda tam iki grup seçilmelidir.

## Örnek sonuç (şirket B, 981 kişi: female + male)

| | |
|---|---|
| Olumlu karar oranı | female %14,0 / male %45,5 |
| Adalet oranı (DI) | 0,307 (%95 aralık 0,235 – 0,390) |
| Doğrudan etki (8 ind-* niteliği sabit) | odds oranı 1,99 (1,22 – 3,23), male / female |

| Yöntem | Doğruluk | DI | EO_farkı |
|---|---|---|---|
| Düzeltme yok | 0,874 | 0,332 | 0,160 |
| ThresholdOptimizer, demographic_parity | 0,812 | 0,866 | 0,427 |
| ThresholdOptimizer, equalized_odds | 0,854 | 0,493 | 0,094 |
| ExponentiatedGradient, demographic_parity | 0,733 | 0,789 | 0,146 |
| ExponentiatedGradient, equalized_odds | 0,829 | 0,504 | 0,118 |

## Sınırlar

- Veri sentetiktir; sonuçlar gerçek bir şirket hakkında bulgu değil, yöntemin göstergesidir.
- Doğrudan etki yalnızca seçilen niteliklere göre hesaplanır. Seçilmeyen nitelikler farkı açıklıyor olabilir; cinsiyetle güçlü ilişkili nitelikleri (proxy) kontrol etmek etkiyi eksik tahmin edebilir.
- Doğruluk, geçmiş kararlara uyum olarak ölçülür. Geçmiş kararlar önyargılıysa doğruluk kaybı gerçek bir kayıp olmayabilir.
- Eşit seçim oranı ve eşit hata oranları genellikle aynı anda sağlanamaz.
- ThresholdOptimizer karar anında grup bilgisini kullanır; bu hukuken tartışmalıdır. ExponentiatedGradient yalnızca eğitimde kullanır.
- Claude yorumu her çalıştırmada değişebilir. Sayıları her zaman rapordaki tablodan doğrula.
- Araç istatistiksel kanıt sunar, hukuki ayrımcılık kararı vermez.