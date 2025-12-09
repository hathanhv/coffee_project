"""
Main Script - Highlands Coffee Customer Segmentation

Pipeline hoàn chỉnh từ preprocessing đến clustering và visualization
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.trainer import ModelTrainer, TrainingConfig
from src.models.evaluator import ClusteringEvaluator
from src.models.tuning import HyperparameterTuner, TuningConfig


def setup_logger():
    """Thiết lập logger cho main script"""
    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    
    return logger


def train_single_model(logger):
    """
    Mode 1: Train một model cụ thể với cấu hình cố định
    """
    logger.info("\n" + "="*80)
    logger.info("MODE 1: TRAIN SINGLE MODEL")
    logger.info("="*80)
    
    # Cấu hình model
    config = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="kmeans",  # Có thể đổi: 'kmeans', 'gmm', 'dbscan', 'hdbscan'
        n_clusters=5,
        model_params={
            "n_init": 20,
            "max_iter": 500
        },
        model_path="results/kmeans_model.pkl"
    )
    
    # Train
    evaluator = ClusteringEvaluator()
    trainer = ModelTrainer(config=config, evaluator=evaluator)
    
    trainer.load_data()
    trainer.train_model()
    metrics = trainer.evaluate()
    
    # Lưu kết quả
    trainer.save_model()
    trainer.save_labels("results/kmeans_labels.csv")
    
    logger.info("\n✅ Single model training completed!")
    logger.info(f"   Model saved: {config.model_path}")
    logger.info(f"   Labels saved: results/kmeans_labels.csv")
    
    return trainer, metrics


def hyperparameter_tuning(logger):
    """
    Mode 2: Grid search tất cả models để tìm best hyperparameters
    """
    logger.info("\n" + "="*80)
    logger.info("MODE 2: HYPERPARAMETER TUNING")
    logger.info("="*80)
    
    tuning_config = TuningConfig(
        data_path="data/processed/encoded_data.csv",
        results_path="results/tuning_results.csv",
        metric_selection="silhouette"  # 'silhouette', 'calinski_harabasz', 'davies_bouldin'
    )
    
    evaluator = ClusteringEvaluator()
    tuner = HyperparameterTuner(config=tuning_config, evaluator=evaluator)
    
    # Run grid search cho tất cả models
    tuner.run_all_models()
    
    # Lưu kết quả
    tuner.save_results()
    tuner.save_best_model_and_df(
        model_path="results/best_model.pkl",
        df_path="results/clustered_data.csv"
    )
    
    logger.info("\n✅ Hyperparameter tuning completed!")
    logger.info(f"   Results saved: {tuning_config.results_path}")
    logger.info(f"   Best model: results/best_model.pkl")
    logger.info(f"   Clustered data: results/clustered_data.csv")
    
    return tuner


def compare_models(logger):
    """
    Mode 3: So sánh nhanh 4 models với cấu hình mặc định
    """
    logger.info("\n" + "="*80)
    logger.info("MODE 3: QUICK MODEL COMPARISON")
    logger.info("="*80)
    
    evaluator = ClusteringEvaluator()
    results = []
    
    # 1. KMeans
    logger.info("\n[1/4] Testing KMeans...")
    config_kmeans = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="kmeans",
        n_clusters=5,
        model_params={"n_init": 20}
    )
    trainer = ModelTrainer(config_kmeans, evaluator)
    trainer.load_data()
    trainer.train_model()
    metrics = trainer.evaluate()
    results.append({"model": "KMeans", **metrics})
    
    # 2. GMM
    logger.info("\n[2/4] Testing GMM...")
    config_gmm = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="gmm",
        n_clusters=5,
        model_params={"covariance_type": "full"}
    )
    trainer = ModelTrainer(config_gmm, evaluator)
    trainer.load_data()
    trainer.train_model()
    metrics = trainer.evaluate()
    results.append({"model": "GMM", **metrics})
    
    # 3. DBSCAN
    logger.info("\n[3/4] Testing DBSCAN...")
    config_dbscan = TrainingConfig(
        data_path="data/processed/encoded_data.csv",
        model_type="dbscan",
        model_params={"eps": 2.0, "min_samples": 10}
    )
    trainer = ModelTrainer(config_dbscan, evaluator)
    trainer.load_data()
    trainer.train_model()
    
    labels = trainer.get_cluster_labels()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters >= 2:
        metrics = trainer.evaluate()
        results.append({"model": "DBSCAN", **metrics})
    else:
        logger.warning(f"  ⚠ DBSCAN only found {n_clusters} cluster(s), skipping evaluation")
    
    # 4. HDBSCAN
    logger.info("\n[4/4] Testing HDBSCAN...")
    try:
        config_hdbscan = TrainingConfig(
            data_path="data/processed/encoded_data.csv",
            model_type="hdbscan",
            model_params={"min_cluster_size": 15, "min_samples": 10}
        )
        trainer = ModelTrainer(config_hdbscan, evaluator)
        trainer.load_data()
        trainer.train_model()
        
        labels = trainer.get_cluster_labels()
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2:
            metrics = trainer.evaluate()
            results.append({"model": "HDBSCAN", **metrics})
        else:
            logger.warning(f"  ⚠ HDBSCAN only found {n_clusters} cluster(s), skipping evaluation")
    except ImportError:
        logger.warning("  ⚠ HDBSCAN not installed, skipping...")
    
    # Hiển thị bảng so sánh
    import pandas as pd
    df_results = pd.DataFrame(results)
    
    logger.info("\n" + "="*80)
    logger.info("📊 MODEL COMPARISON RESULTS")
    logger.info("="*80)
    print(df_results.to_string(index=False))
    
    # Lưu kết quả
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/model_comparison.csv", index=False)
    logger.info(f"\n💾 Comparison saved: results/model_comparison.csv")
    
    return df_results


def main():
    """Main function với menu chọn mode"""
    logger = setup_logger()
    
    logger.info("\n" + "="*80)
    logger.info("🎯 HIGHLANDS COFFEE CUSTOMER SEGMENTATION")
    logger.info("="*80)
    logger.info("\nChọn mode:")
    logger.info("  1. Train single model (nhanh, test model cụ thể)")
    logger.info("  2. Hyperparameter tuning (chậm, tìm best config)")
    logger.info("  3. Quick model comparison (so sánh 4 models)")
    logger.info("  4. Run all (chạy tất cả)")
    
    choice = input("\nNhập lựa chọn (1/2/3/4): ").strip()
    
    if choice == "1":
        train_single_model(logger)
    
    elif choice == "2":
        hyperparameter_tuning(logger)
    
    elif choice == "3":
        compare_models(logger)
    
    elif choice == "4":
        logger.info("\n🚀 Running all modes...")
        train_single_model(logger)
        compare_models(logger)
        hyperparameter_tuning(logger)
    
    else:
        logger.error("❌ Invalid choice!")
        return
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
