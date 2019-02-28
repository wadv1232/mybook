# 分类模型评估

| 真实\预测 | 真 | 假 | 合计 |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 真 | TP 正类判定为正类 | FN 正类判定为负类,"去真" | condition positive  = TP + FN | **真阳性率 TPR** = TP/condition positive &lt;br&gt;又称为 **灵敏度\(Sensitivity\), 召回率\(Recall\)** | 假阴性率 FNR = FN/condition positive  又称为 漏诊率 = 1- 灵敏度 |
| 假 | FP 负类判定为正类,"存伪" | TN 负类判定为负类 | condition negative = FP + TN | 假阳性率 FPR = FP/condition negative  又称为 误诊率 = 1 - 特异度 | **真阴性率 TNR** = TN/condition negtive &lt;br&gt; 又称 **特异度\(specificity\)** |

$$P = TP/（TP+FP）$$

$$R = TP/(TP + FN) $$



## 实例解释 {#实例解释}

下面以医学中[糖尿病](https://www.baidu.com/s?wd=%E7%B3%96%E5%B0%BF%E7%97%85&tn=24004469_oem_dg&rsv_dl=gh_pl_sl_csd)人的筛查为例对敏感度和特异度进行解释。在这个例子中，我们只将病人血糖水平作为判断是否患有糖尿病的指标。下图为正常人和糖尿病患者血糖水平的统计图：

![](/assets/糖尿病血糖水平.png)

