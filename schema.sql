-- ============================================================================
-- OLIST BRAZILIAN E-COMMERCE DATABASE SCHEMA
-- ============================================================================
-- This schema is designed for the Intelligent Marketing Platform
-- Source: Kaggle - Brazilian E-Commerce Public Dataset by Olist
-- ============================================================================

-- Create schema for organizing tables
CREATE SCHEMA IF NOT EXISTS olist;

-- Set search path
SET search_path TO olist, public;

-- ============================================================================
-- 1. CUSTOMERS TABLE
-- ============================================================================
-- Contains customer information and location data
-- customer_id is unique per order, customer_unique_id identifies repeat customers

DROP TABLE IF EXISTS olist.customers CASCADE;

CREATE TABLE olist.customers (
    customer_id VARCHAR(32) PRIMARY KEY,
    customer_unique_id VARCHAR(32) NOT NULL,
    customer_zip_code_prefix VARCHAR(5) NOT NULL,
    customer_city VARCHAR(100) NOT NULL,
    customer_state VARCHAR(2) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_customers_unique_id ON olist.customers(customer_unique_id);
CREATE INDEX idx_customers_state ON olist.customers(customer_state);
CREATE INDEX idx_customers_city ON olist.customers(customer_city);
CREATE INDEX idx_customers_zip ON olist.customers(customer_zip_code_prefix);

COMMENT ON TABLE olist.customers IS 'Customer information with location data. customer_unique_id identifies repeat customers across orders.';
COMMENT ON COLUMN olist.customers.customer_id IS 'Unique identifier for each order-customer combination';
COMMENT ON COLUMN olist.customers.customer_unique_id IS 'Unique identifier for the customer (for repeat purchase tracking)';


-- ============================================================================
-- 2. GEOLOCATION TABLE
-- ============================================================================
-- Brazilian zip codes with latitude/longitude coordinates
-- Used for mapping and distance calculations

DROP TABLE IF EXISTS olist.geolocation CASCADE;

CREATE TABLE olist.geolocation (
    geolocation_id SERIAL PRIMARY KEY,
    geolocation_zip_code_prefix VARCHAR(5) NOT NULL,
    geolocation_lat DECIMAL(10, 8) NOT NULL,
    geolocation_lng DECIMAL(11, 8) NOT NULL,
    geolocation_city VARCHAR(100) NOT NULL,
    geolocation_state VARCHAR(2) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for location queries
CREATE INDEX idx_geolocation_zip ON olist.geolocation(geolocation_zip_code_prefix);
CREATE INDEX idx_geolocation_state ON olist.geolocation(geolocation_state);
CREATE INDEX idx_geolocation_coords ON olist.geolocation(geolocation_lat, geolocation_lng);

COMMENT ON TABLE olist.geolocation IS 'Brazilian zip codes with geographic coordinates for mapping and distance analysis.';


-- ============================================================================
-- 3. SELLERS TABLE
-- ============================================================================
-- Seller information and location

DROP TABLE IF EXISTS olist.sellers CASCADE;

CREATE TABLE olist.sellers (
    seller_id VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(5) NOT NULL,
    seller_city VARCHAR(100) NOT NULL,
    seller_state VARCHAR(2) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_sellers_state ON olist.sellers(seller_state);
CREATE INDEX idx_sellers_city ON olist.sellers(seller_city);
CREATE INDEX idx_sellers_zip ON olist.sellers(seller_zip_code_prefix);

COMMENT ON TABLE olist.sellers IS 'Seller information with location data for fulfillment analysis.';


-- ============================================================================
-- 4. PRODUCT CATEGORY NAME TRANSLATION TABLE
-- ============================================================================
-- Translates Portuguese category names to English

DROP TABLE IF EXISTS olist.product_category_translation CASCADE;

CREATE TABLE olist.product_category_translation (
    product_category_name VARCHAR(100) PRIMARY KEY,
    product_category_name_english VARCHAR(100) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE olist.product_category_translation IS 'Translation table for product categories from Portuguese to English.';


-- ============================================================================
-- 5. PRODUCTS TABLE
-- ============================================================================
-- Product catalog with attributes

DROP TABLE IF EXISTS olist.products CASCADE;

CREATE TABLE olist.products (
    product_id VARCHAR(32) PRIMARY KEY,
    product_category_name VARCHAR(100),
    product_name_length INTEGER,
    product_description_length INTEGER,
    product_photos_qty INTEGER,
    product_weight_g INTEGER,
    product_length_cm INTEGER,
    product_height_cm INTEGER,
    product_width_cm INTEGER,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key to category translation
    CONSTRAINT fk_products_category 
        FOREIGN KEY (product_category_name) 
        REFERENCES olist.product_category_translation(product_category_name)
        ON DELETE SET NULL
);

-- Indexes
CREATE INDEX idx_products_category ON olist.products(product_category_name);
CREATE INDEX idx_products_weight ON olist.products(product_weight_g);

COMMENT ON TABLE olist.products IS 'Product catalog with physical attributes and category information.';
COMMENT ON COLUMN olist.products.product_name_length IS 'Number of characters in the product name';
COMMENT ON COLUMN olist.products.product_description_length IS 'Number of characters in the product description';


-- ============================================================================
-- 6. ORDERS TABLE (Core table)
-- ============================================================================
-- Main orders table - links to all other order-related tables

DROP TABLE IF EXISTS olist.orders CASCADE;

CREATE TABLE olist.orders (
    order_id VARCHAR(32) PRIMARY KEY,
    customer_id VARCHAR(32) NOT NULL,
    order_status VARCHAR(20) NOT NULL,
    order_purchase_timestamp TIMESTAMP NOT NULL,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key
    CONSTRAINT fk_orders_customer 
        FOREIGN KEY (customer_id) 
        REFERENCES olist.customers(customer_id)
        ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX idx_orders_customer ON olist.orders(customer_id);
CREATE INDEX idx_orders_status ON olist.orders(order_status);
CREATE INDEX idx_orders_purchase_date ON olist.orders(order_purchase_timestamp);
CREATE INDEX idx_orders_delivered_date ON olist.orders(order_delivered_customer_date);

-- Partial index for delivered orders
CREATE INDEX idx_orders_delivered ON olist.orders(order_id) 
    WHERE order_status = 'delivered';

COMMENT ON TABLE olist.orders IS 'Core orders table with status and timestamp information.';
COMMENT ON COLUMN olist.orders.order_status IS 'Order status: delivered, shipped, canceled, etc.';


-- ============================================================================
-- 7. ORDER ITEMS TABLE
-- ============================================================================
-- Items within each order (one order can have multiple items)

DROP TABLE IF EXISTS olist.order_items CASCADE;

CREATE TABLE olist.order_items (
    order_id VARCHAR(32) NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id VARCHAR(32) NOT NULL,
    seller_id VARCHAR(32) NOT NULL,
    shipping_limit_date TIMESTAMP NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    freight_value DECIMAL(10, 2) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key (composite)
    PRIMARY KEY (order_id, order_item_id),
    
    -- Foreign keys
    CONSTRAINT fk_order_items_order 
        FOREIGN KEY (order_id) 
        REFERENCES olist.orders(order_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_order_items_product 
        FOREIGN KEY (product_id) 
        REFERENCES olist.products(product_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_order_items_seller 
        FOREIGN KEY (seller_id) 
        REFERENCES olist.sellers(seller_id)
        ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_order_items_order ON olist.order_items(order_id);
CREATE INDEX idx_order_items_product ON olist.order_items(product_id);
CREATE INDEX idx_order_items_seller ON olist.order_items(seller_id);
CREATE INDEX idx_order_items_price ON olist.order_items(price);

COMMENT ON TABLE olist.order_items IS 'Individual items within orders. One order can have multiple items from different sellers.';
COMMENT ON COLUMN olist.order_items.order_item_id IS 'Sequential number for items within an order (1, 2, 3, ...)';
COMMENT ON COLUMN olist.order_items.shipping_limit_date IS 'Deadline for seller to ship the item';


-- ============================================================================
-- 8. ORDER PAYMENTS TABLE
-- ============================================================================
-- Payment information for orders (one order can have multiple payment methods)

DROP TABLE IF EXISTS olist.order_payments CASCADE;

CREATE TABLE olist.order_payments (
    order_id VARCHAR(32) NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_type VARCHAR(20) NOT NULL,
    payment_installments INTEGER NOT NULL,
    payment_value DECIMAL(10, 2) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key (composite)
    PRIMARY KEY (order_id, payment_sequential),
    
    -- Foreign key
    CONSTRAINT fk_order_payments_order 
        FOREIGN KEY (order_id) 
        REFERENCES olist.orders(order_id)
        ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_order_payments_order ON olist.order_payments(order_id);
CREATE INDEX idx_order_payments_type ON olist.order_payments(payment_type);

COMMENT ON TABLE olist.order_payments IS 'Payment details for orders. One order can have multiple payment methods.';
COMMENT ON COLUMN olist.order_payments.payment_sequential IS 'Sequence number for multiple payments on same order';
COMMENT ON COLUMN olist.order_payments.payment_type IS 'Payment method: credit_card, boleto, voucher, debit_card';
COMMENT ON COLUMN olist.order_payments.payment_installments IS 'Number of installments for credit card payments';


-- ============================================================================
-- 9. ORDER REVIEWS TABLE
-- ============================================================================
-- Customer reviews for orders (critical for sentiment analysis)

DROP TABLE IF EXISTS olist.order_reviews CASCADE;

CREATE TABLE olist.order_reviews (
    review_id VARCHAR(32) PRIMARY KEY,
    order_id VARCHAR(32) NOT NULL,
    review_score INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5),
    review_comment_title VARCHAR(500),
    review_comment_message TEXT,
    review_creation_date TIMESTAMP NOT NULL,
    review_answer_timestamp TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key
    CONSTRAINT fk_order_reviews_order 
        FOREIGN KEY (order_id) 
        REFERENCES olist.orders(order_id)
        ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_order_reviews_order ON olist.order_reviews(order_id);
CREATE INDEX idx_order_reviews_score ON olist.order_reviews(review_score);
CREATE INDEX idx_order_reviews_creation_date ON olist.order_reviews(review_creation_date);

-- Full-text search index for review text (for sentiment analysis)
CREATE INDEX idx_order_reviews_message_fts ON olist.order_reviews 
    USING gin(to_tsvector('portuguese', COALESCE(review_comment_message, '')));

COMMENT ON TABLE olist.order_reviews IS 'Customer reviews with ratings and text comments. Key table for sentiment analysis.';
COMMENT ON COLUMN olist.order_reviews.review_score IS 'Rating from 1 (worst) to 5 (best)';
COMMENT ON COLUMN olist.order_reviews.review_comment_message IS 'Free-text review comment (in Portuguese)';

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
