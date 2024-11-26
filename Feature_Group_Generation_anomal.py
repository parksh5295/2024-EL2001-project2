# -*- coding: utf-8 -*-

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# CSV 파일 로드
df = pd.read_csv('./quantized_train_data.csv')

# label=1인 데이터만 필터링
df_label_1 = df[df['label'] == 1]

# 결과 확인
print(df_label_1.head())

# 2. FP-Growth 적용
# feature:value를 하나의 item으로 변환하여 리스트로 만듦
df_label_1_features = df_label_1.drop(columns=['label'])  # 'label'을 제외한 나머지 feature들

# 데이터는 True/False 형태로 변환해야 함
# NaN 값이 아닌 것을 True, NaN 값은 False로 변환
df_label_1_bin = df_label_1_features.notna()

# FP-Growth 적용 (빈발 항목집합 추출)
frequent_itemsets = fpgrowth(df_label_1_bin, min_support=0.7, use_colnames=True)

# 3. 결과로 얻은 frequent_itemsets에서 규칙을 추출
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 4. 결과를 딕셔너리로 변환
# 'antecedents'와 'consequents'의 feature:value를 기준으로 딕셔너리로 구성
rules_dict = {}
for _, row in rules.iterrows():
    antecedents = tuple(row['antecedents'])
    consequents = tuple(row['consequents'])
    # antecedents와 consequents를 합쳐서 하나의 그룹을 구성
    rules_dict[antecedents] = consequents

# 결과 확인
print(rules_dict)
