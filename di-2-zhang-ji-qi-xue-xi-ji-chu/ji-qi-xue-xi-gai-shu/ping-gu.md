# 分类模型评估

| 真实\预测 | 真 | 假 | 合计 |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 真 | TP 正类判定为正类 | FN 正类判定为负类,"去真" | condition positive  = TP + FN | **真阳性率 TPR** = TP/condition positive &lt;br&gt;又称为 **灵敏度\(Sensitivity\), 召回率\(Recall\)** |  |
| 假 | FP 负类判定为正类,"存伪" | TN 负类判定为负类 | condition negative = FP + TN | 假阳性率 FPR = FP/condition negative  又称为 误诊率 = 1 -  | **真阴性率 TNR** = TN/condition negtive &lt;br&gt; 又称 **特异度\(specificity\)** |

$$P = TP/（TP+FP）$$

$$R = TP/(TP + FN) $$

