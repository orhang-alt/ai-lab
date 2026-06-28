# e09 — Optimizer'lar: SGD, Momentum, Adam

## Kavram
Optimizer, backprop gradyanları verdikten sonra parametrelerin nasıl hareket
edeceğine karar verir. Loss fonksiyonu araziyse, optimizer yürüme kuralıdır.

## Hipotez
Aynı eğimli loss üzerinde plain SGD mevcut gradyan yönünde ilerler, Momentum
hızı hatırlayarak yolu yumuşatır, Adam ise birinci ve ikinci moment tahminleriyle
adım ölçeklerini uyarlar.

## Yöntem
1. Tüm optimizer'ları aynı iki parametreden başlat.
2. Aynı bowl-shaped loss'u minimize et.
3. Aynı adım sayısından sonra final parametreleri ve loss'u yazdır.

## Ne gözlemlenmeli?
- SGD çalışır ama learning rate'e hassastır.
- Momentum, hız doğru yöne oturduğunda daha hızlı ilerleyebilir.
- Adam farklı gradyan ölçekleriyle iyi baş eder.

## Sonuç
Optimizer seçimi hedefi değil, parametre uzayında izlenen yolu değiştirir.

