   CREATE DATABASE churn_db;
   
   USE churn_db;
SELECT COUNT(*) FROM customers_raw;

USE churn_db;
SELECT Contract,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers_clean
GROUP BY Contract
ORDER BY churn_pct DESC;


SELECT PaymentMethod,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers_clean
GROUP BY PaymentMethod
ORDER BY churn_pct DESC;

SELECT CASE
         WHEN tenure <= 12 THEN '0-12 months'
         WHEN tenure <= 24 THEN '13-24 months'
         WHEN tenure <= 48 THEN '25-48 months'
         ELSE '49+ months'
       END AS tenure_group,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers_clean
GROUP BY tenure_group
ORDER BY MIN(tenure);

SELECT InternetService,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers_clean
GROUP BY InternetService
ORDER BY churn_pct DESC;

SELECT COUNT(*) FROM customers_features;

SELECT COUNT(*) FROM customer_churn_drivers;
SELECT * FROM shap_importance LIMIT 10;

USE churn_db;
SELECT COUNT(*) FROM churn_predictions;

SELECT risk_level,
       COUNT(*) AS customers,
       ROUND(SUM(MonthlyCharges), 0) AS monthly_revenue
FROM churn_predictions
GROUP BY risk_level;


USE churn_db;
SELECT COUNT(*) FROM retention_recommendations;

SELECT action, COUNT(*) AS customers, ROUND(SUM(revenue_at_risk), 0) AS revenue_at_risk
FROM retention_recommendations
GROUP BY action
ORDER BY customers DESC;