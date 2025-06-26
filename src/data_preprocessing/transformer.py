import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

class DataTransformer:
    def __init__(self, use_spacy: bool = False):
        if not use_spacy:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoders_map = {} # For reverse mapping label encoders
        self.label_encoder = LabelEncoder()
        self.multi_label_binarizers = MultiLabelBinarizer() # For knowledge sources
        self.use_spacy = use_spacy
        if use_spacy:
            self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text: str) -> str:
        """Tokenizes, removes stop words, and lemmatizes the input text."""
        if not isinstance(text, str):
            return ""
        
        if not self.use_spacy:
            tokens = word_tokenize(text.lower())
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
            filtered_tokens = [token for token in lemmatized_tokens if token not in self.stop_words]

        if self.use_spacy:
            doc = self.nlp(text.lower())
            filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return ' '.join(filtered_tokens)

    def apply_text_preprocessing(self, df: pd.DataFrame, text_column: str = 'message') -> pd.DataFrame:
        """Applies text preprocessing to a column. For example, 'message' column"""
        print(f"Applying text preprocessing to column '{text_column}'...")
        if text_column in df.columns:
            df[f'{text_column}_processed'] = df[text_column].apply(self.preprocess_text)
        else:
            print(f"Warning: Text column '{text_column}' not found.")
        return df
    
    def encode_categorical_variables(self, df : pd.DataFrame, columns_to_encode: list[str]= None) -> pd.DataFrame:
        """Encodes categorical columns to numerical representations using Label Encoding."""
        if columns_to_encode is None:
            columns_to_encode = ['agent', 'sentiment', 'turn_rating', 'config'] # Setting default categorical columns

        for col in columns_to_encode:
            if col in df.columns and df[col].dtype.name in ['category', 'object']:
                df[f"{col}_encoded"] = self.label_encoder.fit_transform(df[col])
                self.label_encoders_map[col] = self.label_encoder
                print(f"Encoded column {col}. Unique values: {df[col].nunique()}")
            elif col not in df.columns:
                print(f"Warning: {col} not in the data")
        return df
    
    def encode_multilabel_knowledge_source(self, df : pd.DataFrame, column_name: str = 'knowledge_source')-> pd.DataFrame:
        """Encodes the multi-label 'knowledge_source' column using MultiLabelBinarizer."""
        print(f"Encoding multi-label column '{column_name}'...")
        if column_name in df.columns:
            ks_encoded = self.multi_label_binarizers.fit_transform(df[column_name])
            ks_df = pd.DataFrame(ks_encoded, columns=[f"ks_{cls}" for cls in self.multi_label_binarizers.classes_], index=df.index)
            df = pd.concat([df, ks_df], axis=1)
            print(f"Encoded '{column_name}' into {len(self.multi_label_binarizers.classes_)} binary columns.")
        else:
            print(f"Warning: Column '{column_name}' not found for multi-label encoding.")
        return df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all transformation steps."""
        df = self.apply_text_preprocessing(df.copy(), text_column='message')
        df = self.encode_categorical_variables(df.copy())
        df = self.encode_multilabel_knowledge_source(df.copy())
        print("Data transformation complete.")
        return df