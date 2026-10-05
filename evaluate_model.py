import os

import pandas as pd
import numpy as np

TARGET = 'is_high_risk'
THRESHOLD = 0.5  # Поріг, вище якого прогнозуємо клас 1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))


def predict_proba(df, model_df):
    """Імовірність класу 1 для сирих рядків df за збереженою моделлю (theta, mean, std)."""
    features = model_df['feature'].tolist()[1:]
    # Кодуємо так само, як при навчанні; reindex гарантує той самий набір і порядок колонок,
    # навіть якщо якоїсь категорії у вибірці немає
    X = pd.get_dummies(df.drop(columns=[TARGET, 'row_id'], errors='ignore')).astype(float)
    X = X.reindex(columns=features, fill_value=0.0).values
    # Масштабуємо параметрами навчальної вибірки
    X = (X - model_df['mean'].values[1:]) / model_df['std'].values[1:]
    X = np.c_[np.ones(X.shape[0]), X]
    return sigmoid(X @ model_df['theta'].values)


def compute_metrics(y, p, threshold=THRESHOLD):
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    eps = 1e-15
    log_loss = -np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))

    return {
        'samples': len(y),
        'accuracy': (tp + tn) / len(y),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'log_loss': log_loss,
        # Точність "наївної" моделі, яка завжди прогнозує найчастіший клас
        'baseline_accuracy': max(np.mean(y), 1 - np.mean(y)),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def evaluate(results_dir='results'):
    model_df = pd.read_csv(os.path.join(results_dir, 'model_params.csv'))

    metrics = {}
    for split in ['train', 'test']:
        df = pd.read_csv(os.path.join(results_dir, f'{split}_set.csv'))
        y = df[TARGET].values
        p = predict_proba(df, model_df)
        metrics[split] = compute_metrics(y, p)

        if split == 'test':
            # Прогноз для кожного рядка тестової вибірки
            pred_df = df.assign(probability=p, predicted=(p >= THRESHOLD).astype(int))
            pred_df['correct'] = (pred_df['predicted'] == pred_df[TARGET]).astype(int)
            pred_df.to_csv(os.path.join(results_dir, 'test_predictions.csv'), index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index_label='metric')

    print("\n=== ОЦІНКА МОДЕЛІ ===")
    print(f"{'Метрика':<25} {'train':>10} {'test':>10}")
    for name, row in metrics_df.iterrows():
        if name in ('samples', 'tp', 'tn', 'fp', 'fn'):
            print(f"{name:.<25} {int(row['train']):>10d} {int(row['test']):>10d}")
        else:
            print(f"{name:.<25} {row['train']:>10.4f} {row['test']:>10.4f}")

    t = metrics['test']
    print("\n=== МАТРИЦЯ ПОМИЛОК (Test set) ===")
    print(f"{'':<20} {'прогноз 0':>10} {'прогноз 1':>10}")
    print(f"{'факт 0':<20} {t['tn']:>10d} {t['fp']:>10d}")
    print(f"{'факт 1':<20} {t['fn']:>10d} {t['tp']:>10d}")

    print(f"\nФайли збережено в '{results_dir}/'.")
    return metrics_df


if __name__ == '__main__':
    evaluate()
