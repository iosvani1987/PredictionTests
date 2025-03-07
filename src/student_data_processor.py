import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class StudentDataPreprocessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.student_df = None

    def load_data(self):
        self.student_df = pd.read_csv(self.input_path)

    def calculate_score(self):
        score_columns = [column for column in self.student_df.columns if column.endswith('score')]
        self.student_df['score'] = round(self.student_df[score_columns].sum(axis=1) / 30)
        self.student_df.drop(columns=score_columns, inplace=True)

    def encode_categorical(self, columns: list[str]):        
        for column in columns:
            # One-hot encode 'gender'
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(self.student_df[[column]])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]))
            self.student_df = pd.concat([self.student_df, encoded_df], axis=1)
            self.student_df.drop(columns=[column], inplace=True)
    
    def ordinal_encode(self, column: str, ordinal_categories: list[str]):
        # Ordinal encode 'parental_level_of_education'
        ordinal_encoder = OrdinalEncoder(categories=[ordinal_categories])
        encoded_data = ordinal_encoder.fit_transform(self.student_df[['parental_level_of_education']])
        df_encoded = pd.DataFrame(encoded_data, columns=['parental_level_of_education_encoded'])
        self.student_df = pd.concat([self.student_df, df_encoded], axis=1)
        self.student_df.drop('parental_level_of_education', axis=1, inplace=True)

    def save_data(self):
        self.student_df.to_csv(self.output_path, index=False)

    def preprocess(self):
        self.load_data()
        self.calculate_score()        
        
        columns_to_encode = ['gender', 'race_ethnicity', 'lunch', 'test_preparation_course']
        self.encode_categorical(columns_to_encode)  
        
        ordinal_categories = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
        self.ordinal_encode('parental_level_of_education', ordinal_categories)      
        print(self.student_df.head())
        
        self.save_data()
        

