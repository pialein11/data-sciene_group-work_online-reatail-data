# Online Shoppers Purchasing Intention Dataset

## Introduction and problem statement
The **Online Shoppers Purchasing Intention Dataset** is a multivariate collection of features capturing the real-time clickstream data and behavior patterns of visitors on an e-commerce website. Donated in August 2018, the dataset aggregates 12,330 unique sessions collected over a one-year period. Crucially, the dataset was designed so that each session belongs to a different user, explicitly mitigating any localized bias toward specific promotional campaigns, user profiles, distinct calendar periods, or special days.

The primary e-commerce problem statement is to predict user intent—specifically whether an online session will culminate in a financial transaction (**Revenue** generation). Framed as a classification or clustering task in the introductory work by *Sakar et al. (2019)*, models developed on this data allow businesses to understand buying indicators in real time. This enables live site optimizations, personalized marketing interventions, and dynamic user experience tailoring before the shopper abandons their cart.

---

## Contained attributes
The dataset contains **12,330 instances** characterized by **17 predictive features** (split into 10 numerical and 7 categorical/binary attributes). These features track page category views, duration metrics, Google Analytics web behavior flags, and temporal or demographic identifiers:

### Session Activity & Duration Features
* **Administrative**: The number of administrative pages (e.g., account management, checkout steps) visited by the user during the session.
* **Administrative_Duration**: Total time (in seconds) the user spent on administrative pages.
* **Informational**: The number of informational pages (e.g., about us, contact info, shipping terms) visited by the user during the session.
* **Informational_Duration**: Total time (in seconds) the user spent on informational pages.
* **ProductRelated**: The number of product-specific pages (e.g., item details, catalogs) visited by the user during the session.
* **ProductRelated_Duration**: Total time (in seconds) the user spent on product pages.

### Google Analytics & Web Metrics
* **BounceRate**: The percentage of visitors who entered the site through a specific page and left ("bounced") without triggering any subsequent interactions in that session.
* **ExitRate**: The percentage of all pageviews to a specific page that marked the absolute final interaction in the user's session.
* **PageValue**: The average value of a web page determined by Google Analytics, calculated based on pages visited by a user right before completing an e-commerce transaction.

### Temporal & Demographic Identifiers
* **SpecialDay**: A numeric indicator ($0.0$ to $1.0$) representing the proximity of the session to a significant calendar event (e.g., Valentine's Day, Mother's Day). The value scales based on shipping lead times; for example, around Valentine's Day, it takes non-zero values between February 2nd and 12th, peaking at $1.0$ on February 8th.
* **Month**: The month of the year during which the session took place (Categorical).
* **OperatingSystems**: An encoded integer indicating the user's operating system.
* **Browser**: An encoded integer indicating the user's web browser type.
* **Region**: An encoded integer indicating the geographic location/region of the user.
* **TrafficType**: An encoded integer mapping how the user arrived at the site (e.g., direct, organic search, paid ads).
* **VisitorType**: A categorical string stating whether the user is a `Returning_Visitor`, a `New_Visitor`, or `Other`.
* **Weekend**: A boolean flag indicating whether the session occurred over the weekend (True/False).

*Note: This dataset contains no missing values.*

---

## Potential targets
The definitive target class label for this dataset is **`Revenue`**.

* **`Revenue` (Target Label)**: A binary indicator representing whether the online shopping session concluded with a finalized purchase transaction.
    * **False (Negative Class)**: The session did **not** end with a transaction. This class represents **84.5% (10,422 sessions)** of the dataset.
    * **True (Positive Class)**: The session successfully ended in a transaction. This class represents **15.5% (1,908 sessions)** of the dataset.

Because the data displays a stark class imbalance (nearly 5.5 to 1 negative-to-positive ratio), models trained on this target must rely on evaluation metrics like Precision, Recall, F1-Score, and Precision-Recall Area Under the Curve (PR-AUC) rather than raw accuracy alone.