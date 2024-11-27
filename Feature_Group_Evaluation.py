import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

component_data = pd.read_csv('./feature_contribution.csv', header=None, names=['feature', 'component'])

top_5_features = component_data.sort_values(by='component', ascending=False).head(5)['feature'].tolist()
print("Top 5 Features:", top_5_features)

df = pd.read_csv('./clustering_data.csv')

df_top_5 = df[top_5_features + ['cluster']]

cluster_variance = df_top_5.groupby('cluster').var()
print("Cluster Variance:\n", cluster_variance)

label_encoder = LabelEncoder()
df_top_5['cluster_encoded'] = label_encoder.fit_transform(df_top_5['cluster'])

from sklearn.metrics import precision_score, recall_score

precision_list = []
recall_list = []

for feature in top_5_features:
    threshold = df_top_5[feature].median()  # 중간값을 기준으로 임계값 설정
    df_top_5['predicted_cluster'] = (df_top_5[feature] > threshold).astype(int)

    precision = precision_score(df_top_5['cluster_encoded'], df_top_5['predicted_cluster'], average='micro', zero_division=0)
    recall = recall_score(df_top_5['cluster_encoded'], df_top_5['predicted_cluster'], average='micro', zero_division=0)

    precision_list.append(precision)
    recall_list.append(recall)

    print(f"Feature: {feature}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}\n")

print("Precision List:", precision_list)
print("Recall List:", recall_list)

results_df = pd.DataFrame({
    'Feature': top_5_features,
    'Precision': precision_list,
    'Recall': recall_list
})

results_df.to_csv('./feature_evaluation_results.csv', index=False)