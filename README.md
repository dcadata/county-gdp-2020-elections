# County-level analysis of GDP and 2020 election results

## GDP

Via [BEA - GDP by County, Metro, and Other Areas](https://www.bea.gov/data/gdp/gdp-county-metro-and-other-areas). From [this page](https://apps.bea.gov/regional/downloadzip.cfm), I downloaded the dataset "CAGDP1: GDP Summary by County and MSA". County data was used in this analysis, and GDP is in current-dollar GDP (thousands of current dollars).

## Population

Population estimates from [this page](https://www.census.gov/data/tables/time-series/demo/popest/2010s-counties-total.html): "Datasets" > "Population, Population Change, and Estimated Components of Population Change: April 1, 2010 to July 1, 2019 (CO-EST2019-alldata)"

## State FIPS

FIPS code for each state from [this link at census.gov](https://www.census.gov/geographies/reference-files/2017/demo/popest/2017-fips.html)

## Election Results

Unofficial county-level election results via Politico / Associated Press feed. (I originally did this analysis before official results were available.)

Scraped for each state FIPS code using the following URL formats:

- counties (FIPS-to-name): https://www.politico.com/2020-statewide-metadata/{STATE_FIPS_CODE}/county-names.meta.json
- candidates: https://www.politico.com/2020-statewide-metadata/{STATE_FIPS_CODE}/potus.meta.json
- county-level results: https://www.politico.com/2020-statewide-results/{STATE_FIPS_CODE}/potus-counties.json

---

## To-Do

- use official election results instead
- publish Tableau Public workbook/visualizations
- summarize outputs
