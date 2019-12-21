# interpretable-machine-learning

# psa-analysis

The purpose of the research is to see whether interpretable machine learning models can be helpful in assessing pre-trial risk, in comparison to the Arnold foundation's current methods. Our goal will be to take their pre-trial risk measure (PSA), create our own risk assessment tools using interpretable machine learning methods, and do an empirical comparison. 

# coding notes 

For the time being, the cleaned Broward County data (Table_construction.Rdata) has been directly copy pasted from the Age of Unfairness repository.  

# compute-psa.Rmd

Computes the PSA, computes recidivism outcomes, saves this to a new .Rdata file stored in `\broward-data`. 

We take "prior sentence to incarceration" to mean prison because jail is short-term. Additionally, everybody in the Broward dataset has gone to jail (because they have COMPAS evaluations), so incarceration would not be a useful feature if we counted jail. 
We will take "screening date" as current, and any offenses which occurred up to 30 days before this screening date as current offense (partially because 30 days seems reasonable to me and also to stay consistent with ProPublica's definition of recidivism, which defines a current case as within 30 days of the screening date).
