#!/usr/bin/env python3
"""
OLIST DATA LOADER
=================
This script loads the Olist Brazilian E-Commerce dataset from CSV files into PostgreSQL.

Usage:
    python load_olist_data.py --data-dir /path/to/csv/files --db-url postgresql://user:pass@host:port/dbname

Requirements:
    pip install pandas psycopg2-binary sqlalchemy python-dotenv tqdm

Author: Marketing AI Platform
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from io import StringIO
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('olist_data_load.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Mapping of CSV files to table names and their load order (respecting foreign keys)
CSV_TABLE_MAPPING = {
    # Order matters for foreign key constraints!
    1: {
        'file': 'olist_customers_dataset.csv',
        'table': 'customers',
        'columns': {
            'customer_id': 'customer_id',
            'customer_unique_id': 'customer_unique_id',
            'customer_zip_code_prefix': 'customer_zip_code_prefix',
            'customer_city': 'customer_city',
            'customer_state': 'customer_state'
        }
    },
    2: {
        'file': 'olist_geolocation_dataset.csv',
        'table': 'geolocation',
        'columns': {
            'geolocation_zip_code_prefix': 'geolocation_zip_code_prefix',
            'geolocation_lat': 'geolocation_lat',
            'geolocation_lng': 'geolocation_lng',
            'geolocation_city': 'geolocation_city',
            'geolocation_state': 'geolocation_state'
        }
    },
    3: {
        'file': 'olist_sellers_dataset.csv',
        'table': 'sellers',
        'columns': {
            'seller_id': 'seller_id',
            'seller_zip_code_prefix': 'seller_zip_code_prefix',
            'seller_city': 'seller_city',
            'seller_state': 'seller_state'
        }
    },
    4: {
        'file': 'product_category_name_translation.csv',
        'table': 'product_category_translation',
        'columns': {
            'product_category_name': 'product_category_name',
            'product_category_name_english': 'product_category_name_english'
        }
    },
    5: {
        'file': 'olist_products_dataset.csv',
        'table': 'products',
        'columns': {
            'product_id': 'product_id',
            'product_category_name': 'product_category_name',
            'product_name_lenght': 'product_name_length',  # Note: typo in source
            'product_description_lenght': 'product_description_length',  # Note: typo in source
            'product_photos_qty': 'product_photos_qty',
            'product_weight_g': 'product_weight_g',
            'product_length_cm': 'product_length_cm',
            'product_height_cm': 'product_height_cm',
            'product_width_cm': 'product_width_cm'
        }
    },
    6: {
        'file': 'olist_orders_dataset.csv',
        'table': 'orders',
        'columns': {
            'order_id': 'order_id',
            'customer_id': 'customer_id',
            'order_status': 'order_status',
            'order_purchase_timestamp': 'order_purchase_timestamp',
            'order_approved_at': 'order_approved_at',
            'order_delivered_carrier_date': 'order_delivered_carrier_date',
            'order_delivered_customer_date': 'order_delivered_customer_date',
            'order_estimated_delivery_date': 'order_estimated_delivery_date'
        },
        'date_columns': [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
    },
    7: {
        'file': 'olist_order_items_dataset.csv',
        'table': 'order_items',
        'columns': {
            'order_id': 'order_id',
            'order_item_id': 'order_item_id',
            'product_id': 'product_id',
            'seller_id': 'seller_id',
            'shipping_limit_date': 'shipping_limit_date',
            'price': 'price',
            'freight_value': 'freight_value'
        },
        'date_columns': ['shipping_limit_date']
    },
    8: {
        'file': 'olist_order_payments_dataset.csv',
        'table': 'order_payments',
        'columns': {
            'order_id': 'order_id',
            'payment_sequential': 'payment_sequential',
            'payment_type': 'payment_type',
            'payment_installments': 'payment_installments',
            'payment_value': 'payment_value'
        }
    },
    9: {
        'file': 'olist_order_reviews_dataset.csv',
        'table': 'order_reviews',
        'columns': {
            'review_id': 'review_id',
            'order_id': 'order_id',
            'review_score': 'review_score',
            'review_comment_title': 'review_comment_title',
            'review_comment_message': 'review_comment_message',
            'review_creation_date': 'review_creation_date',
            'review_answer_timestamp': 'review_answer_timestamp'
        },
        'date_columns': ['review_creation_date', 'review_answer_timestamp']
    }
}


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

class OlistDataLoader:
    """Handles loading Olist CSV data into PostgreSQL."""
    
    def __init__(self, db_url: str, schema: str = 'olist'):
        """
        Initialize the data loader.
        
        Args:
            db_url: PostgreSQL connection string
            schema: Database schema name
        """
        self.db_url = db_url
        self.schema = schema
        self.engine = None
        self.stats = {
            'tables_loaded': 0,
            'total_rows': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.engine = create_engine(self.db_url, echo=False)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to PostgreSQL")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_schema(self) -> bool:
        """Create the olist schema if it doesn't exist."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
                conn.commit()
            logger.info(f"Schema '{self.schema}' ready")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to create schema: {e}")
            return False
    
    def run_schema_sql(self, schema_file: str) -> bool:
        """Execute the schema SQL file."""
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
            
            with self.engine.connect() as conn:
                for stmt in tqdm(statements, desc="Creating schema"):
                    if stmt and not stmt.startswith('--'):
                        try:
                            conn.execute(text(stmt))
                        except SQLAlchemyError as e:
                            # Log but continue (some statements might fail if objects exist)
                            logger.warning(f"Statement warning: {str(e)[:100]}")
                conn.commit()
            
            logger.info("Schema SQL executed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to execute schema SQL: {e}")
            return False
    
    def disable_foreign_keys(self):
        """Temporarily disable foreign key checks for faster loading."""
        with self.engine.connect() as conn:
            conn.execute(text("SET session_replication_role = 'replica';"))
            conn.commit()
        logger.info("Foreign key checks disabled")
    
    def enable_foreign_keys(self):
        """Re-enable foreign key checks."""
        with self.engine.connect() as conn:
            conn.execute(text("SET session_replication_role = 'origin';"))
            conn.commit()
        logger.info("Foreign key checks enabled")
    
    def truncate_table(self, table_name: str):
        """Truncate a table before loading."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"TRUNCATE TABLE {self.schema}.{table_name} CASCADE"))
                conn.commit()
            logger.info(f"Truncated table {table_name}")
        except SQLAlchemyError as e:
            logger.warning(f"Could not truncate {table_name}: {e}")
    
    def load_with_copy(self, df: pd.DataFrame, table_name: str):
        """Use PostgreSQL COPY for fastest bulk loading."""
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N')
        buffer.seek(0)
        
        # Get column names from DataFrame to specify in COPY command
        columns = ', '.join(df.columns)
        
        conn = self.engine.raw_connection()
        try:
            with conn.cursor() as cur:
                cur.copy_expert(
                    f"COPY {self.schema}.{table_name} ({columns}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')",
                    buffer
                )
            conn.commit()
        finally:
            conn.close()
    
    def load_csv_to_table(
        self,
        csv_path: str,
        table_name: str,
        column_mapping: Dict[str, str],
        date_columns: list = None,
        chunk_size: int = 10000,
        truncate_first: bool = True
    ) -> Tuple[bool, int]:
        """
        Load a CSV file into a PostgreSQL table.
        
        Args:
            csv_path: Path to the CSV file
            table_name: Target table name
            column_mapping: Dict mapping CSV columns to table columns
            date_columns: List of columns to parse as dates
            chunk_size: Number of rows per batch
            truncate_first: Whether to truncate the table before loading
            
        Returns:
            Tuple of (success, rows_loaded)
        """
        # Use os.path for cross-platform compatibility
        csv_path = os.path.abspath(os.path.expanduser(csv_path))
        
        if not os.path.isfile(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False, 0
        
        try:
            # Read CSV with proper handling
            logger.info(f"Reading {csv_path}...")
            
            # Determine which columns to parse as dates
            parse_dates = date_columns if date_columns else False
            
            df = pd.read_csv(
                csv_path,
                parse_dates=parse_dates,
                low_memory=False,
                encoding='utf-8'
            )
            
            # Rename columns according to mapping
            df = df.rename(columns=column_mapping)
            
            # Keep only mapped columns that exist
            valid_columns = [col for col in column_mapping.values() if col in df.columns]
            df = df[valid_columns]
            
            # Convert float columns that should be integers
            int_columns = ['product_name_length', 'product_description_length', 
                          'product_photos_qty', 'product_weight_g', 'product_length_cm',
                          'product_height_cm', 'product_width_cm', 'payment_installments',
                          'payment_sequential', 'review_score', 'order_item_id']
            for col in int_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
            # Handle NaN values
            df = df.where(pd.notnull(df), None)
            
            # Truncate if requested
            if truncate_first:
                self.truncate_table(table_name)
            
            # Load data using COPY (much faster than INSERT)
            total_rows = len(df)
            logger.info(f"Loading {total_rows:,} rows into {table_name} using COPY...")

            if table_name == "order_reviews": # handling the duplicate records
                df = df.drop_duplicates(subset=['review_id'], keep='first')
            
            self.load_with_copy(df, table_name)
            
            logger.info(f"Successfully loaded {total_rows:,} rows into {table_name}")
            return True, total_rows
            
        except Exception as e:
            logger.error(f"Failed to load {csv_path} into {table_name}: {e}")
            self.stats['errors'].append({
                'table': table_name,
                'file': csv_path,
                'error': str(e)
            })
            return False, 0
    
    def load_all_data(
        self,
        data_dir: str,
        truncate_first: bool = True,
        skip_existing: bool = False
    ) -> bool:
        """
        Load all Olist CSV files into the database.
        
        Args:
            data_dir: Directory containing CSV files
            truncate_first: Whether to truncate tables before loading
            skip_existing: Skip tables that already have data
            
        Returns:
            True if all tables loaded successfully
        """
        self.stats['start_time'] = datetime.now()
        # Use os.path for cross-platform compatibility
        data_path = os.path.abspath(os.path.expanduser(data_dir))
        
        if not os.path.isdir(data_path):
            logger.error(f"Data directory not found: {data_dir}")
            return False
        
        # Disable foreign keys for faster loading
        self.disable_foreign_keys()
        
        success = True
        
        # Load tables in order
        for order in sorted(CSV_TABLE_MAPPING.keys()):
            config = CSV_TABLE_MAPPING[order]
            csv_file = os.path.join(data_path, config['file'])
            table_name = config['table']
            column_mapping = config['columns']
            date_columns = config.get('date_columns', None)
            
            if not os.path.isfile(csv_file):
                logger.warning(f"CSV file not found, skipping: {csv_file}")
                continue
            
            # Check if table has data (skip if requested)
            if skip_existing:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {self.schema}.{table_name}")
                    )
                    count = result.scalar()
                    if count > 0:
                        logger.info(f"Skipping {table_name} (already has {count:,} rows)")
                        continue
            
            # Load the table
            try:
                loaded, rows = self.load_csv_to_table(
                    csv_path=str(csv_file),
                    table_name=table_name,
                    column_mapping=column_mapping,
                    date_columns=date_columns,
                    truncate_first=truncate_first
                )
                
                if loaded:
                    self.stats['tables_loaded'] += 1
                    self.stats['total_rows'] += rows
                else:
                    success = False
                    
            except Exception as e:
                # Log the error but continue with other files
                self.stats['tables_failed'] = self.stats.get('tables_failed', 0) + 1
                print(f"Error loading {table_name} from {csv_file}: {e}")
                # Optionally store failed tables for reporting
                if 'failed_tables' not in self.stats:
                    self.stats['failed_tables'] = []
                self.stats['failed_tables'].append({
                    'table': table_name,
                    'file': str(csv_file),
                    'error': str(e)
                })
            
            if loaded:
                self.stats['tables_loaded'] += 1
                self.stats['total_rows'] += rows
            else:
                success = False
        
        # Re-enable foreign keys
        self.enable_foreign_keys()
        
        self.stats['end_time'] = datetime.now()
        return success
    
    def verify_data(self) -> Dict:
        """Verify loaded data by counting rows in each table."""
        verification = {}
        
        with self.engine.connect() as conn:
            for order in sorted(CSV_TABLE_MAPPING.keys()):
                table_name = CSV_TABLE_MAPPING[order]['table']
                try:
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {self.schema}.{table_name}")
                    )
                    count = result.scalar()
                    verification[table_name] = count
                except SQLAlchemyError:
                    verification[table_name] = 'ERROR'
        
        return verification
    
    def print_summary(self):
        """Print loading summary."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print("\n" + "=" * 60)
        print("OLIST DATA LOADING SUMMARY")
        print("=" * 60)
        print(f"Tables loaded:    {self.stats['tables_loaded']}")
        print(f"Total rows:       {self.stats['total_rows']:,}")
        print(f"Duration:         {duration:.2f} seconds")
        print(f"Errors:           {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\nErrors:")
            for err in self.stats['errors']:
                print(f"  - {err['table']}: {err['error'][:50]}...")
        
        print("\nRow counts per table:")
        verification = self.verify_data()
        for table, count in verification.items():
            print(f"  {table:35} {count:>10,}" if isinstance(count, int) else f"  {table:35} {count}")
        
        print("=" * 60)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_csv_files(data_dir: str) -> Tuple[list, list]:
    """
    Validate that all expected CSV files exist.
    
    Returns:
        Tuple of (found_files, missing_files)
    """
    # Convert to absolute path and resolve any symlinks
    data_path = os.path.abspath(os.path.expanduser(data_dir))
    found = []
    missing = []
    print(data_path)

    # Debug: Check if directory exists
    if not os.path.isdir(data_path):
        print("here")
        logger.error(f"Data directory does not exist: {data_path}")
        return found, list(config['file'] for config in CSV_TABLE_MAPPING.values())
    
    # List actual files in directory for debugging
    try:
        actual_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        print(actual_files)
        logger.info(f"Found {len(actual_files)} CSV files in directory: {actual_files[:3]}...")
    except Exception as e:
        logger.error(f"Error listing directory: {e}")
    
    for config in CSV_TABLE_MAPPING.values():
        # Use os.path.join for cross-platform compatibility
        csv_file = os.path.join(data_path, config['file'])
        print(csv_file)
        
        if os.path.isfile(csv_file):
            found.append(config['file'])
        else:
            missing.append(config['file'])
    
    return found, missing

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Load Olist Brazilian E-Commerce data into PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load data with explicit connection string
    python load_olist_data.py --data-dir ./data --db-url postgresql://user:pass@localhost:5432/olist
    
    # Load data using environment variables
    export DATABASE_URL="postgresql://user:pass@localhost:5432/olist"
    python load_olist_data.py --data-dir ./data
    
    # Validate CSV files without loading
    python load_olist_data.py --data-dir ./data --validate-only
    
    # Create schema only (no data loading)
    python load_olist_data.py --db-url postgresql://... --schema-only --schema-file schema.sql
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing Olist CSV files'
    )
    
    parser.add_argument(
        '--db-url',
        type=str,
        help='PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/dbname)'
    )
    
    parser.add_argument(
        '--schema-file',
        type=str,
        default='schema.sql',
        help='Path to schema SQL file (default: schema.sql)'
    )
    
    parser.add_argument(
        '--schema-only',
        action='store_true',
        help='Only create schema, do not load data'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate CSV files, do not load'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip tables that already have data'
    )
    
    parser.add_argument(
        '--no-truncate',
        action='store_true',
        help='Do not truncate tables before loading (append mode)'
    )
    
    parser.add_argument(
        '--create-env-template',
        action='store_true',
        help='Create .env.template file and exit'
    )
    
    args = parser.parse_args()
        
    # Get database URL from args or environment
    db_url = args.db_url or os.getenv('DATABASE_URL')
    
    # If no URL provided, try to construct from individual env vars
    if not db_url:
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'olist_marketing')
        user = os.getenv('POSTGRES_USER', 'rumasinha')
        #password = os.getenv('POSTGRES_PASSWORD')
        
        #if password:
        db_url = f"postgresql+psycopg2://{user}@{host}:{port}/{db}"
    
    # Handle validate-only mode
    if args.validate_only:
        if not args.data_dir:
            print("Error: --data-dir is required for validation")
            return 1
        
        found, missing = validate_csv_files(args.data_dir)
        
        print("\nCSV File Validation")
        print("=" * 40)
        print(f"Found:   {len(found)} files")
        print(f"Missing: {len(missing)} files")
        
        if found:
            print("\nFound files:")
            for f in found:
                print(f"  ✓ {f}")
        
        if missing:
            print("\nMissing files:")
            for f in missing:
                print(f"  ✗ {f}")
            return 1
        
        print("\nAll files found!")
        return 0
    
    # Validate required arguments
    if not db_url:
        print("Error: Database URL required. Use --db-url or set DATABASE_URL environment variable.")
        print("Run with --create-env-template to create a template .env file.")
        return 1
    
    print(db_url)

    # Initialize loader
    loader = OlistDataLoader(db_url)
    
    # Connect to database
    if not loader.connect():
        return 1
    
    # Handle schema-only mode
    if args.schema_only:
        if not os.path.exists(args.schema_file):
            print(f"Error: Schema file not found: {args.schema_file}")
            return 1
        
        if loader.run_schema_sql(args.schema_file):
            print("Schema created successfully!")
            return 0
        return 1
    
    # Full data loading
    if not args.data_dir:
        print("Error: --data-dir is required for data loading")
        return 1
    
    # Validate CSV files first
    found, missing = validate_csv_files(args.data_dir)
    if missing:
        print(f"Warning: {len(missing)} CSV files not found:")
        for f in missing:
            print(f"  - {f}")
        print("Continuing with available files...")
    
    # Run schema SQL if file exists
    if os.path.exists(args.schema_file):
        logger.info(f"Running schema from {args.schema_file}")
        loader.run_schema_sql(args.schema_file)
    else:
        logger.warning(f"Schema file not found: {args.schema_file}, assuming schema exists")
    
    # Load data
    success = loader.load_all_data(
        data_dir=args.data_dir,
        truncate_first=not args.no_truncate,
        skip_existing=args.skip_existing
    )
    
    # Print summary
    loader.print_summary()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
