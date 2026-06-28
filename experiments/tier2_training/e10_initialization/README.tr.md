# e10 — Initialization ve Sinyal Ölçeği

## Kavram
Initialization, öğrenme başlamadan önce sinyallerin ve gradyanların nasıl
göründüğünü belirler. Kötü başlangıç nöronları aynılaştırabilir, aktivasyonları
doyurabilir veya sinyali sıfıra indirebilir.

## Hipotez
Sıfır initialization simetriyi bozar; büyük normal initialization tanh'ı doyurur;
Xavier tanh sinyal ölçeğini daha dengeli tutar; He ise ReLU sinyal ölçeğini daha
dengeli tutar.

## Yöntem
1. Random veriyi birkaç random katmandan geçir.
2. Katman katman aktivasyon mean/std değerlerini karşılaştır.
3. Tüm sıfır satırların neden aynı nöronlar oluşturduğunu göster.

## Ne gözlemlenmeli?
- Sıfır ağırlıklar nöronlara aynı çıktıyı verir.
- Büyük normal ağırlıklar tanh'ı saturasyona iter.
- Xavier ve He sihirli sayı değil; farklı aktivasyonlar için ölçeği koruma
  tahminleridir.

## Sonuç
Initialization, vanishing/exploding sinyale karşı ilk savunmadır.

