from student_data_processor import StudentDataPreprocessor
from model_trainer import ModelTrainer


def main():
    input_path = 'data/raw/study_performance.csv'
    output_path = 'data/processed/study_performance_processed.csv'
    processor = StudentDataPreprocessor(input_path, output_path)
    processor.preprocess()

    # Train the model Random Forest
    model_trainer_rf = ModelTrainer(output_path, model_type='RF')
    model_trainer_rf.train_models()
    mse_rf = model_trainer_rf.evaluate_models()

    # Train the model Logistic Regression
    model_trainer_lr = ModelTrainer(output_path, model_type='LR')
    model_trainer_lr.train_models()
    mse_lr = model_trainer_lr.evaluate_models()

    # Display the sample predictions
    print(f"Random Forest MSE: {mse_rf}")
    print(f"Logistic Regression MSE: {mse_lr}")

if __name__ == "__main__":
    main()