# KDD Report

## Selection
The selected dataset is the UCI Online Retail transaction table. It contains 541,909 rows and 8 columns covering December 1, 2010 through December 9, 2011. The table contains 25,900 invoices, 4,070 stock codes, and 4,372 distinct customer identifiers. The discovery objective is to identify interpretable customer segments and product co-occurrence rules that could support exploratory merchandising decisions.

The selection stage retains the complete raw table for audit, while the analytical population will require valid customer IDs, valid invoice/product identifiers, positive quantity and price, and non-cancelled invoices. The executed inventory found 135,080 rows with missing customer IDs, 9,288 cancellation rows, 10,624 non-positive quantities, and 2,517 non-positive prices. These counts are selection evidence, not yet final deletion decisions.

## Preprocessing
The preprocessing script applies deterministic rules in an auditable order: remove cancelled invoices, remove rows without a customer ID, remove non-positive quantities or prices, and remove invalid dates. After these rules, 397,884 rows remain, representing 18,532 invoices and 4,338 customers. The audit records 134,697 missing-customer rows, 9,288 cancellation rows, and 40 additional non-positive-price rows as the first applicable reason. The raw dataset remains unchanged and the clean table is a derived artifact.

## Transformation
The transformation stage creates line revenue as `Quantity × UnitPrice`, then aggregates transactions to customer-level RFM features. Recency is measured from December 10, 2011, the day after the latest retained transaction; frequency is the number of distinct invoices; and monetary value is total revenue. The resulting RFM table contains 4,338 customers. A binary invoice-by-product basket matrix was also created for the 100 most frequent stock codes, producing 15,219 invoice rows and 100 product columns. The top-100 restriction keeps association mining computationally manageable and is documented rather than hidden.

## Data mining
The RFM features were log-transformed and standardized before KMeans clustering. Candidate values from k=2 through k=6 were compared using silhouette score. k=2 had the highest silhouette score (0.4329), so it was selected. The two resulting groups were a larger, less recent/less frequent/less monetary group of 2,671 customers and a smaller, more recent/frequent/high-value group of 1,667 customers.

For product discovery, the binary top-100 basket matrix was mined with pairwise support, confidence, and lift. The implementation retained pairs with at least 20 co-occurring invoices and produced 4,771 candidate rules before ranking. The strongest example connected the Green, Pink, and Roses Regency teacup products, with lift values above 15. These are co-occurrence patterns, not causal recommendations.

## Interpretation and evaluation
Support, confidence, and lift are interpreted together. Cluster size and RFM means are reported so that a segment is not described only by an algorithmic label. Rules are ranked by lift and support and enriched with product descriptions. Sensitivity is represented by comparing candidate cluster counts and by requiring a minimum co-occurrence count. The analysis does not establish causal demand, profit impact, or future stability; those claims would require temporal validation and an operational experiment.
