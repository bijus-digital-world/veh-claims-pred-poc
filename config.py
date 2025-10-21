"""
Configuration Management for Nissan Telematics POC

Centralized configuration with environment variable support.
Load from .env file or use defaults.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AWSConfig:
    """AWS-related configuration"""
    region: str = "us-east-1"
    s3_bucket: str = "veh-poc-207567760844-us-east-1"
    place_index_name: str = "NissanPlaceIndex"
    use_s3: bool = True
    
    @classmethod
    def from_env(cls):
        return cls(
            region=os.getenv("AWS_REGION", "us-east-1"),
            s3_bucket=os.getenv("S3_BUCKET", "veh-poc-207567760844-us-east-1"),
            place_index_name=os.getenv("PLACE_INDEX_NAME", "NissanPlaceIndex"),
            use_s3=os.getenv("USE_S3", "true").lower() == "true"
        )


@dataclass
class PathsConfig:
    """File and directory paths"""
    # Base directories
    base_dir: Path = Path(__file__).parent
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    vector_dir: Path = field(default_factory=lambda: Path("vector_store"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    images_dir: Path = field(default_factory=lambda: Path("images"))
    
    # Data files
    local_data_file: str = "data/vehicle_claims_extended.csv"
    s3_data_key: str = "data/vehicle_claims_extended.csv"
    
    # Model files
    model_path: str = "models/claim_rate_model.joblib"
    
    # Log files
    inference_log_local: str = "inference_log.csv"
    inference_log_s3_key: str = "logs/inference_log.csv"
    chat_log_file: str = "logs/chat_history.csv"
    
    # Vector store files
    faiss_index_path: Path = field(default_factory=lambda: Path("vector_store/historical_data_index.faiss"))
    embeddings_path: Path = field(default_factory=lambda: Path("vector_store/historical_data_embs.npy"))
    metadata_path: Path = field(default_factory=lambda: Path("vector_store/historical_data_meta.npy"))
    metadata_json_path: Path = field(default_factory=lambda: Path("vector_store/historical_data_meta.json"))
    
    # Image files
    maintenance_icon: str = "images/maintenance_icon.svg"
    nissan_logo: str = "images/nissan_logo.svg"
    
    @classmethod
    def from_env(cls):
        base = Path(os.getenv("BASE_DIR", Path(__file__).parent))
        return cls(
            base_dir=base,
            data_dir=base / os.getenv("DATA_DIR", "data"),
            models_dir=base / os.getenv("MODELS_DIR", "models"),
            vector_dir=base / os.getenv("VECTOR_DIR", "vector_store"),
            logs_dir=base / os.getenv("LOGS_DIR", "logs"),
            images_dir=base / os.getenv("IMAGES_DIR", "images"),
            local_data_file=os.getenv("LOCAL_DATA_FILE", "data/vehicle_claims_extended.csv"),
            s3_data_key=os.getenv("S3_DATA_KEY", "data/vehicle_claims_extended.csv"),
            model_path=os.getenv("MODEL_PATH", "models/claim_rate_model.joblib"),
            inference_log_local=os.getenv("INFERENCE_LOG_LOCAL", "inference_log.csv"),
            inference_log_s3_key=os.getenv("INFERENCE_LOG_S3_KEY", "logs/inference_log.csv"),
            chat_log_file=os.getenv("CHAT_LOG_FILE", "logs/chat_history.csv")
        )
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in [self.data_dir, self.models_dir, self.vector_dir, self.logs_dir, self.images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """ML Model configuration"""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Bedrock LLM settings
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    bedrock_max_tokens: int = 320
    bedrock_temperature: float = 0.18
    
    # Prediction thresholds
    default_threshold_pct: int = 80
    threshold_options: List[int] = field(default_factory=lambda: [50, 60, 70, 75, 80, 85, 90, 95, 98, 99])
    
    @classmethod
    def from_env(cls):
        return cls(
            embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            bedrock_max_tokens=int(os.getenv("BEDROCK_MAX_TOKENS", "320")),
            bedrock_temperature=float(os.getenv("BEDROCK_TEMPERATURE", "0.18")),
            default_threshold_pct=int(os.getenv("DEFAULT_THRESHOLD_PCT", "80"))
        )


@dataclass
class UIConfig:
    """UI/UX configuration"""
    page_title: str = "Nissan - Vehicle Predictive Insights (POC)"
    page_icon: str = "images/maintenance_icon.svg"
    layout: str = "wide"
    initial_sidebar_state: str = "collapsed"
    
    # Feature flags
    is_poc: bool = False
    show_repair_cost: bool = False
    
    # Refresh intervals (in seconds)
    refresh_intervals: dict = field(default_factory=lambda: {
        "15s": 15,
        "30s": 30,
        "1m": 60,
        "5m": 300,
        "15m": 900
    })
    default_refresh_interval: str = "15m"
    
    @classmethod
    def from_env(cls):
        return cls(
            page_title=os.getenv("PAGE_TITLE", "Nissan - Vehicle Predictive Insights (POC)"),
            page_icon=os.getenv("PAGE_ICON", "images/maintenance_icon.svg"),
            is_poc=os.getenv("IS_POC", "false").lower() == "true",
            show_repair_cost=os.getenv("SHOW_REPAIR_COST", "false").lower() == "true"
        )


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Required columns (historical data may have buckets, inference uses continuous)
    required_columns: set = field(default_factory=lambda: {
        "model", "primary_failed_part", "mileage_bucket", "age_bucket",
        "date", "claims_count", "repairs_count", "recalls_count"
    })
    
    # Note: Inference rows use 'mileage' and 'age' (continuous values)
    # Historical data uses 'mileage_bucket' and 'age_bucket' for visualization
    
    # Categorical values
    models: List[str] = field(default_factory=lambda: ["Leaf", "Ariya", "Sentra"])
    mileage_buckets: List[str] = field(default_factory=lambda: ["0-10k", "10-30k", "30-60k", "60k+"])
    age_buckets: List[str] = field(default_factory=lambda: ["<1yr", "1-3yr", "3-5yr", "5+yr"])
    
    # Dealer search
    location_prob_threshold: float = 0.5
    max_dealer_distance_km: float = 32.1869  # 20 miles
    dealer_search_top_n: int = 3
    
    # RAG/Search
    rag_top_k: int = 6
    tfidf_max_features: int = 6000
    
    @classmethod
    def from_env(cls):
        return cls(
            location_prob_threshold=float(os.getenv("LOCATION_PROB_THRESHOLD", "0.5")),
            max_dealer_distance_km=float(os.getenv("MAX_DEALER_DISTANCE_KM", "32.1869")),
            dealer_search_top_n=int(os.getenv("DEALER_SEARCH_TOP_N", "3")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "6")),
            tfidf_max_features=int(os.getenv("TFIDF_MAX_FEATURES", "6000"))
        )


@dataclass
class ColorConfig:
    """Nissan brand colors and styling"""
    nissan_red: str = "#c3002f"
    nissan_dark: str = "#0b0f13"
    nissan_gray: str = "#94a3b8"
    nissan_gold: str = "#f59e0b"
    
    # Heatmap color scale
    heatmap_scale: List[str] = field(default_factory=lambda: [
        "#ffffff", "#ffecec", "#ffd1d1", "#ff9b9b",
        "#ff7070", "#c3002f", "#7a0000", "#000000"
    ])
    
    # Risk colors
    risk_colors: dict = field(default_factory=lambda: {
        "Low": "#16a34a",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    })


@dataclass
class RiskConfig:
    """Risk level thresholds"""
    high_threshold: float = 75.0
    medium_threshold: float = 50.0
    low_threshold: float = 0.0
    
    @classmethod
    def from_env(cls):
        return cls(
            high_threshold=float(os.getenv("RISK_HIGH_THRESHOLD", "75.0")),
            medium_threshold=float(os.getenv("RISK_MEDIUM_THRESHOLD", "50.0")),
            low_threshold=float(os.getenv("RISK_LOW_THRESHOLD", "0.0"))
        )


@dataclass
class Config:
    """Main configuration class combining all config sections"""
    aws: AWSConfig = field(default_factory=AWSConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    colors: ColorConfig = field(default_factory=ColorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            aws=AWSConfig.from_env(),
            paths=PathsConfig.from_env(),
            model=ModelConfig.from_env(),
            ui=UIConfig.from_env(),
            data=DataConfig.from_env(),
            colors=ColorConfig(),
            risk=RiskConfig.from_env(),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def initialize(self):
        """Initialize configuration (create directories, etc.)"""
        self.paths.ensure_directories()
        
        if self.debug:
            print(f"[Config] Environment: {self.environment}")
            print(f"[Config] AWS Region: {self.aws.region}")
            print(f"[Config] S3 Bucket: {self.aws.s3_bucket}")
            print(f"[Config] Using S3: {self.aws.use_s3}")


# Load environment variables from .env file if it exists
def load_env_file(env_file: str = ".env"):
    """Load environment variables from .env file"""
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value


# Initialize global config
load_env_file()
config = Config.from_env()
config.initialize()


# Convenience exports for backward compatibility
USE_S3 = config.aws.use_s3
S3_BUCKET = config.aws.s3_bucket
AWS_REGION = config.aws.region
MODEL_PATH = config.paths.model_path
MODELS = config.data.models
MILEAGE_BUCKETS = config.data.mileage_buckets
AGE_BUCKETS = config.data.age_buckets
NISSAN_RED = config.colors.nissan_red
NISSAN_DARK = config.colors.nissan_dark
NISSAN_GRAY = config.colors.nissan_gray
NISSAN_GOLD = config.colors.nissan_gold
NISSAN_HEATMAP_SCALE = config.colors.heatmap_scale
EMBED_MODEL_NAME = config.model.embedding_model_name

