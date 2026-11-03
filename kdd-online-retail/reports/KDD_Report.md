# KDD Report

## Selection
Select transaction records with valid invoice, product, quantity, price, and customer identifiers.

## Preprocessing
Remove cancellations and non-positive quantities/prices, parse dates, and document records removed.

## Transformation
Create revenue, recency, frequency, monetary value, and basket-level item representations.

## Data mining
Use standardized RFM features for customer clustering and Apriori/FP-growth-style association analysis for baskets.

## Interpretation and evaluation
Rank rules by support, confidence, and lift, remove redundant low-support rules, inspect cluster sizes, and test whether findings are robust to support thresholds. Correlation or co-occurrence is not treated as causation.
