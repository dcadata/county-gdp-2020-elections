"""

# Data Sources

## GDP

Via [BEA - GDP by County, Metro, and Other Areas](https://www.bea.gov/data/gdp/gdp-county-metro-and-other-areas). From [this page](https://apps.bea.gov/regional/downloadzip.cfm), I downloaded the dataset "CAGDP1: GDP Summary by County and MSA". County data was used in this analysis, and GDP is in current-dollar GDP (thousands of current dollars).

## Population

Population estimates from [this page](https://www.census.gov/data/tables/time-series/demo/popest/2010s-counties-total.html): "Datasets" > "Population, Population Change, and Estimated Components of Population Change: April 1, 2010 to July 1, 2019 (CO-EST2019-alldata)"

## State FIPS

FIPS code for each state from [this link at census.gov](https://www.census.gov/geographies/reference-files/2017/demo/popest/2017-fips.html)

## Election Results

Unofficial county-level election results via Politico / Associated Press feed. (I originally did this analysis before the official results were available.)

Scraped for each state FIPS code using the following URL formats:

- counties (FIPS-to-name): https://www.politico.com/2020-statewide-metadata/{STATE_FIPS_CODE}/county-names.meta.json
- candidates: https://www.politico.com/2020-statewide-metadata/{STATE_FIPS_CODE}/potus.meta.json
- county-level results: https://www.politico.com/2020-statewide-results/{STATE_FIPS_CODE}/potus-counties.json

"""

from json import dump
from time import sleep
import numpy as np
import pandas as pd
from requests import get

_DATA_DIR = 'data/'
_SCRAPED_DIR = _DATA_DIR + 'scraped/'
_VOTE_RESULTS_FILEPATH = _DATA_DIR + 'vote_results.csv'
_VOTE_SUMMARY_FILEPATH = _DATA_DIR + 'vote_summary.csv'


class FilenamePrefixes:
    counties = 'counties_'
    candidates = 'candidates_'
    results = 'results_'


class SingleStateScraper:
    # if I were doing this again, I would download the raw files and then parse them separately

    _counties_url = 'https://www.politico.com/2020-statewide-metadata/{}/county-names.meta.json'
    _candidates_url = 'https://www.politico.com/2020-statewide-metadata/{}/potus.meta.json'
    _results_url = 'https://www.politico.com/2020-statewide-results/{}/potus-counties.json'

    def __init__(self, **kwargs):
        self._fips_num = kwargs.get('fips_num')

    def scrape_state(self):
        self._scrape_counties()
        self._scrape_candidates()
        self._scrape_results()

    def _scrape_counties(self):
        data = self._make_request(self._counties_url.format(self._fips_num))
        counties = pd.DataFrame(data.items(), columns=['fips_county', 'name_county'])
        counties.to_csv(_SCRAPED_DIR + f'{FilenamePrefixes.counties}{self._fips_num}.csv', index=False)

    def _scrape_candidates(self):
        data = self._make_request(self._candidates_url.format(self._fips_num))

        result = []
        try:
            for c in data.get('candidates'):
                result.append(self._get_candidate_info_helper(c, data))
        except AttributeError:  # result is a list instead of dict
            # use only statewide candidate IDs (Maine and Nebraska)
            for c in data[0].get('candidates'):
                result.append(self._get_candidate_info_helper(c, data[0]))

        candidates = pd.DataFrame(result)
        candidates = _normalize_column_names(candidates)
        candidates = candidates.loc[:, ['fips_state', 'candidateid', 'shortname', 'fullname']]
        candidates.to_csv(_SCRAPED_DIR + f'{FilenamePrefixes.candidates}{self._fips_num}.csv', index=False)

    def _scrape_results(self):
        data = self._make_request(self._results_url.format(self._fips_num))

        result = []
        for race in data.get('races'):
            for c in race.get('candidates'):
                info = {
                    'fips_state': race.get('stateFips'),
                    'fips_county': race.get('countyFips'),
                }
                info.update(c)
                info['lastupdated'] = data.get('lastUpdated')
                result.append(info)

        results = pd.DataFrame(result)
        results = _normalize_column_names(results)
        results.to_csv(_SCRAPED_DIR + f'{FilenamePrefixes.results}{self._fips_num}.csv', index=False)

    @staticmethod
    def _make_request(url):
        data = get(url, timeout=20).json()
        sleep(1)
        return data

    @staticmethod
    def _get_candidate_info_helper(c, subdict):
        info = {'fips_state': subdict.get('stateFips')}
        info.update(c)
        return info


class MultiStateScraper:
    def __init__(self, **kwargs):
        self._fips_nums = kwargs.get('fips_nums')
        self.has_exceptions = None

    def scrape_multi_state(self):
        exceptions = []

        for num in self._fips_nums:
            try:
                SingleStateScraper(fips_num=num).scrape_state()
            except Exception as exc:
                exceptions.append({'fips_num': num, 'exception': str(exc)})

        if exceptions:
            self.has_exceptions = True
            pd.DataFrame(exceptions).to_csv(_DATA_DIR + 'scraper_exceptions.csv', index=False)

    def merge_multi_state(self):
        dfs = {
            FilenamePrefixes.counties: [],
            FilenamePrefixes.candidates: [],
            FilenamePrefixes.results: [],
        }
        dtypes = {'fips_state': str, 'fips_county': str, 'candidateid': str}

        for filename_prefix in dfs:
            dfs[filename_prefix] = pd.concat([
                pd.read_csv(_SCRAPED_DIR + f'{filename_prefix}{i}.csv', dtype=dtypes) for i in self._fips_nums])

        results = (
            dfs[FilenamePrefixes.results]
                .merge(dfs[FilenamePrefixes.counties], on='fips_county', suffixes=('', '_county'))
                .merge(dfs[FilenamePrefixes.candidates], on=['fips_state', 'candidateid'], suffixes=('', '_candidate'))
        )
        results.to_csv(_VOTE_RESULTS_FILEPATH, index=False)


class DataManager:
    _county_gdp_filepath = _DATA_DIR + 'sources/CAGDP1__ALL_AREAS_2001_2019.csv'
    _state_fips_filepath = _DATA_DIR + 'sources/state-geocodes-v2017.xlsx'
    _pop_est_filepath = _DATA_DIR + 'sources/co-est2019-alldata.csv'
    _latest_year = 2019
    _latest_year_str = str(_latest_year)

    @property
    def _results_from_disk(self):
        return pd.read_csv(_VOTE_RESULTS_FILEPATH, dtype={
            'fips_state': str,
            'fips_county': str,
            'candidateid': str,
            'lastupdated': str,
            'name_county': str,
            'shortname': str,
            'fullname': str,
            'vote': int,
        })

    def refresh_vote_summary(self):
        df = (
            self._major_party_results
                .merge(self.states, on='fips_state')
                .merge(self._vote_totals, on=['fips_state', 'fips_county'], suffixes=('', '_total'))
                .merge(self._winners, on=['fips_state', 'fips_county'], suffixes=('', '_candidate'))
        )
        df['vote_share'] = df.vote / df.vote_total
        df = df.rename(columns={
            'candidateid': 'candidate_id',
            'shortname': 'candidate_shortname',
            'fullname': 'candidate_name',
            'lastupdated': 'last_updated'
        }).drop(columns=['vote'])
        df.r_margin = df.r_margin / df.vote_total
        df.to_csv(_VOTE_SUMMARY_FILEPATH, index=False)

    @property
    def _vote_summary(self):
        dtypes = {
            'fips_state': str,
            'fips_county': str,
            'name_state': str,
            'name_county': str,
            'vote_total': int,
            'candidate_id': str,
            'candidate_shortname': str,
            'candidate_name': str,
            'vote_d': int,
            'vote_r': int,
            'r_winner': int,
            'vote_share': float,
            'r_margin': float,
            'last_updated': str,
        }
        return pd.read_csv(_VOTE_SUMMARY_FILEPATH, usecols=dtypes.keys(), dtype=dtypes)

    @property
    def _county_gdp_base(self):
        dtypes = {
            'GeoFIPS': str,
            'GeoName': str,
            'Region': str,
            'TableName': str,
            'LineCode': float,
            'IndustryClassification': str,
            'Description': str,
            'Unit': str,
        }
        for year in range(2001, 2019 + 1):
            dtypes[str(year)] = str

        df = pd.read_csv(self._county_gdp_filepath, dtype=dtypes, encoding='windows-1252')
        df = _normalize_column_names(df)
        df.geofips = df.geofips.apply(lambda x: x.replace('"', '').strip())
        df = df[df.description == 'Current-dollar GDP (thousands of current dollars)'].copy()
        return df

    @property
    def _county_gdp(self):
        df = self._county_gdp_base.copy()
        years = [str(yr) for yr in range(2001, self._latest_year + 1)]
        for year in years:
            df[year] = df[year].fillna(0).apply(lambda x: x.replace('(NA)', '0')).apply(int)
            df = df[df[year] > 0].copy()

        df = df.loc[:, ['geofips', 'geoname', *years]].rename(columns={'geofips': 'fips_county'})
        return df

    @property
    def _county_gdp_latest_year(self):
        county_gdp_base = self._county_gdp_base.copy()
        county_gdp = county_gdp_base.loc[:, [*list(county_gdp_base.columns)[:8], self._latest_year_str]]
        county_gdp[self._latest_year_str] = county_gdp[self._latest_year_str].fillna(0).apply(lambda x: x.replace(
            '(NA)', '0')).apply(int)
        county_gdp = county_gdp.loc[:, ['geofips', self._latest_year_str]].rename(columns={
            'geofips': 'fips_county', self._latest_year_str: 'gdp'})
        return county_gdp

    @property
    def _population(self):
        dtypes = {'STATE': str, 'COUNTY': str, 'POPESTIMATE2019': int}  # use 2019 population to go with 2019 GDP
        df = pd.read_csv(self._pop_est_filepath, usecols=dtypes.keys(), dtype=dtypes, encoding='windows-1252')
        df = _normalize_column_names(df)
        df.county = df.state + df.county
        df = df.drop(columns='state').rename(columns={'county': 'fips_county', 'popestimate2019': 'pop_est'})
        return df

    @property
    def states(self):
        df = pd.read_excel(self._state_fips_filepath)
        columns = list(df.loc[4, :])
        df = df.loc[5:, :].copy()
        df.columns = columns
        df = _normalize_column_names(df)
        df = df[(df.division != '0') & (df['state (fips)'] != '00')].copy()
        df.name = df.name.apply(lambda x: x.strip())
        df = df.loc[:, ['state (fips)', 'name']].rename(columns={'state (fips)': 'fips_state', 'name': 'name_state'})
        return df

    @property
    def _county_gdp_with_population(self):
        df = self._county_gdp_latest_year.merge(self._population, on='fips_county')
        df['gdp_per_capita'] = df.gdp / df.pop_est
        return df

    @property
    def _major_party_results(self):
        res = self._results_from_disk.copy()
        return res[res.shortname.isin({'Biden', 'Trump'})]

    @property
    def _vote_totals(self):
        return self._major_party_results.groupby(by=['fips_state', 'fips_county'], as_index=False).vote.sum()

    @property
    def _winners(self):
        winners = self._major_party_results.groupby(by=[
            'fips_state', 'fips_county', 'shortname'], as_index=False).vote.sum()
        vote_d = winners[winners.shortname == 'Biden'].drop(columns='shortname')
        vote_r = winners[winners.shortname == 'Trump'].drop(columns='shortname')
        winners = vote_d.merge(vote_r, on=['fips_state', 'fips_county'], suffixes=('_d', '_r'))
        winners['r_winner'] = [1 if r > d else 0 for d, r in zip(winners.vote_d, winners.vote_r)]
        winners['r_margin'] = [r - d for d, r in zip(winners.vote_d, winners.vote_r)]
        return winners

    @property
    def _national_gdp(self):
        df = self._county_gdp_base.copy()
        return int(
            df.loc[df.geoname == 'United States', self._latest_year_str].reset_index().loc[0, self._latest_year_str])


class OutputGenerator(DataManager):
    _years_elapsed = (2, 4, 8, 10)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = {}

    @property
    def vote_summary_with_gdp(self):
        df = self._vote_summary.merge(self._county_gdp_with_population, on='fips_county')
        df['gdp_weighted_by_vote_share'] = df.gdp * df.vote_share
        df['name_county_with_state'] = [f'{c} ({s})' for c, s in zip(df.name_county, df.name_state)]
        return df

    @property
    def _county_gdp_growth(self):
        growth = self._county_gdp.copy()
        for elapsed in self._years_elapsed:
            start_yr = str(self._latest_year - elapsed)
            growth[f'{elapsed}y_growth'] = growth[self._latest_year_str] - growth[start_yr]
            growth[f'{elapsed}y_growth_pct'] = growth[f'{elapsed}y_growth'] / growth[start_yr]
        growth = growth.drop(columns=[f'{yr}' for yr in range(2001, self._latest_year + 1)] + ['geoname'])
        return growth

    @property
    def county_gdp_growth_with_vote_share(self):
        growth = self._county_gdp_growth.merge(self._vote_summary, on='fips_county')
        return growth

    def run(self):
        self._calc_gdp_plain()
        self._calc_gdp_weighted()
        self._calc_gdp_per_capita()
        self._calc_gdp_growth()

    def _calc_gdp_plain(self):
        func = lambda x: round(self._filter_gdp_prep_on_r_winner(x)['gdp'].sum() / self._national_gdp, 3)
        self.output['gdp'] = self._create_output(
            func(0), func(1), description='Proportion of GDP accounted for by counties won by each major party')

    def _calc_gdp_weighted(self):
        df = self.vote_summary_with_gdp.copy()
        func = lambda name: round(
            df[df.candidate_shortname == name].gdp_weighted_by_vote_share.sum() / self._national_gdp, 3)
        self.output['gdp_weighted'] = self._create_output(
            func('Biden'), func('Trump'),
            description='Proportion of GDP accounted for by counties won by each major party - weighted by vote share'
                        ' instead of using win/loss'
        )

    def _calc_gdp_per_capita(self):
        select_field = lambda x: self._filter_gdp_prep_on_r_winner(x)['gdp_per_capita']
        func_median = lambda x: round(np.median(select_field(x)), 2)
        func_mean = lambda x: round(np.mean(select_field(x)), 2)
        func_standard_deviation = lambda x: round(np.std(select_field(x), ddof=0), 2)
        self.output['gdp_per_capita'] = {
            'description': 'GDP per capita across counties won by each major party',
            'd_median': func_median(0), 'r_median': func_median(1),
            'd_mean': func_mean(0), 'r_mean': func_mean(1),
            'd_standard_deviation': func_standard_deviation(0), 'r_standard_deviation': func_standard_deviation(1),
        }

    def _calc_gdp_growth(self):
        vs = self.county_gdp_growth_with_vote_share.copy()
        biden = vs[vs.candidate_shortname == 'Biden'].copy()
        output = {
            'description': 'Simple correlation at county-level of GDP growth over specified time frame to D vote share'}
        for elapsed in self._years_elapsed:
            output[f'{elapsed}y_growth_pct'] = round(biden[f'{elapsed}y_growth_pct'].corr(biden.vote_share), 3)

        self.output['gdp_growth'] = output

    def _filter_gdp_prep_on_r_winner(self, x):
        df = self._vote_summary.loc[:, ['fips_county', 'r_winner']].drop_duplicates().merge(
            self._county_gdp_with_population, on='fips_county')
        return df[df.r_winner == x]

    @staticmethod
    def _create_output(d, r, **kwargs):
        description = kwargs.get('description')
        statement = kwargs.get('statement')

        output = {}
        if description:
            output['description'] = description

        output.update({'d': d, 'r': r})

        if statement:
            output['statement'] = statement

        return output


def _normalize_column_names(df):
    return df.rename(columns=dict((col, col.strip().lower()) for col in df.columns))


def run_scraper():
    scraper = MultiStateScraper(fips_nums=DataManager().states.fips_state)
    scraper.scrape_multi_state()
    if scraper.has_exceptions:
        print('Scraper encountered exceptions.')
    else:
        scraper.merge_multi_state()


def regenerate_results():
    MultiStateScraper(fips_nums=DataManager().states.fips_state).merge_multi_state()
    output = OutputGenerator()
    output.refresh_vote_summary()
    output.vote_summary_with_gdp.to_csv(_VOTE_SUMMARY_FILEPATH + '.with_gdp.csv', index=False)


def generate_output():
    analyzer = OutputGenerator()
    analyzer.run()
    dump(analyzer.output, open(_DATA_DIR + 'output.json', 'w'))


if __name__ == '__main__':
    regenerate_results()
    generate_output()
