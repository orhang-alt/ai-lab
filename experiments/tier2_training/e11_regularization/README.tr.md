# e11 — Regularization ve Overfitting

## Kavram
Overfitting, modelin genel örüntü yerine training set'in gürültüsünü öğrenmesidir.
Regularization daha basit çözümleri tercih eden bir baskı ekler.

## Hipotez
Yüksek dereceli bir polinom train error'ı çok düşürürken validation error yüksek
kalabilir. L2 regularization train error'ı biraz artırır ama validation error'ı
düşürür ve ağırlıkları küçültür.

## Yöntem
1. Düzgün bir eğriden gürültülü örnekler üret.
2. L2 olmadan ve L2 ile yüksek dereceli polinom fit et.
3. Train MSE, validation MSE ve weight norm değerlerini karşılaştır.

## Ne gözlemlenmeli?
- Regularization olmayan model gürültüyü kovalar.
- L2 daha küçük ağırlıkları tercih eder.
- Genelleme training loss ile değil, validation loss ile değerlendirilir.

## Sonuç
Hedef "en düşük training loss" değil; yeni veride işe yarayan davranıştır.

