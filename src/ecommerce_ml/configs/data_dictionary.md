# Olist Data Dictionary

## Tables

| CSV | Purpose | Primary key |
|---|---|---|
| `olist_customers_dataset.csv` | Customer location and IDs | `customer_id` |
| `olist_orders_dataset.csv` | Order status and timestamps | `order_id` |
| `olist_order_items_dataset.csv` | Products within orders | `order_id`, `order_item_id` |
| `olist_order_payments_dataset.csv` | Payment records | `order_id`, `payment_sequential` |
| `olist_order_reviews_dataset.csv` | Reviews and scores | `review_id` |
| `olist_products_dataset.csv` | Product attributes | `product_id` |
| `olist_sellers_dataset.csv` | Seller information | `seller_id` |
| `olist_geolocation_dataset.csv` | Postal-code coordinates | No assumed unique key |
| `product_category_name_translation.csv` | Portuguese-to-English categories | `product_category_name` |

## Important Relationships

- `orders.customer_id` → `customers.customer_id`
- `order_items.order_id` → `orders.order_id`
- `order_items.product_id` → `products.product_id`
- `order_items.seller_id` → `sellers.seller_id`
- `payments.order_id` → `orders.order_id`
- `reviews.order_id` → `orders.order_id`

## Data Modeling Notes

- `customer_id` identifies a customer record; `customer_unique_id` can link repeat purchases by the same person.
- Order items use a composite key: (`order_id`, `order_item_id`).
- Payments and reviews can have multiple records per order.
- Geolocation may contain repeated postal-code prefixes.
- Missing delivery timestamps may be valid for canceled or undelivered orders.

Validate all keys and relationships against the actual CSV files before enforcing database constraints.
