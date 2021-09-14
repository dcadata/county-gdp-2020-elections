import numpy as np
import pandas as pd

_DATA_DIR = 'data/'


class ElectionResultsParser:
    _election_results_fp = _DATA_DIR + 'sources/countypres_2000-2020.csv'

    @property
    def _raw_results(self):
        dtypes = {
            'year': int,
            'county_fips': str,
            'state': str,
            'state_po': str,
            'county_name': str,
            'party': str,
            'candidatevotes': float,
            'totalvotes': float,
            'mode': str,
        }
        df = pd.read_csv(self._election_results_fp, usecols=dtypes.keys(), dtype=dtypes)
        return df

    @property
    def _with_corrected_fips(self):
        df = self._raw_results.copy()
        df = df[df.year == 2020].drop(columns=['year'])
        df.loc[df.state == 'DISTRICT OF COLUMBIA', 'county_fips'] = '11001'  # DC has no fips for 2020 in this dataset
        df = df[df.county_fips.notna()].copy()
        df = df[~df.county_fips.str.endswith('000')].copy()  # fips ending with '000' are state-level
        return df

    @property
    def _modes(self):
        """Modes (early vote, election day, etc.) are separated in this dataset"""
        allmodes = self._with_corrected_fips[['county_fips', 'party', 'candidatevotes', 'totalvotes', 'mode']]
        totalmodes = allmodes[allmodes['mode'] == 'TOTAL'].drop(columns=['mode'])
        nototalmodes = allmodes[~allmodes.county_fips.isin(totalmodes.county_fips)].copy()
        nototalmodes_grouped = nototalmodes.groupby(by=['county_fips', 'party'], as_index=False).agg({
            'candidatevotes': sum, 'totalvotes': sum})
        modes = totalmodes.append(nototalmodes_grouped)
        return modes

    @property
    def _with_combined_modes(self):
        df = self._with_corrected_fips.drop(columns=['candidatevotes', 'totalvotes', 'mode']).drop_duplicates()
        df = df.merge(self._modes, on=['county_fips', 'party'])
        return df

    @property
    def _with_total_votes(self):
        df = self._with_combined_modes.copy()
        df.candidatevotes = df.candidatevotes.fillna(0.0)
        for county_fips in df.loc[df.totalvotes.isna(), :].county_fips:
            totalvotes = df.loc[df.county_fips == county_fips, :].candidatevotes.sum()
            df.loc[df.county_fips == county_fips, 'totalvotes'] = totalvotes
        df.candidatevotes = df.candidatevotes.apply(int)
        df.totalvotes = df.totalvotes.apply(int)
        return df

    @property
    def _major_parties(self):
        df = self._with_total_votes.copy()
        df.loc[df.party == 'DEMOCRAT', 'candidate_shortname'] = 'Biden'
        df.loc[df.party == 'REPUBLICAN', 'candidate_shortname'] = 'Trump'
        df = df[df.candidate_shortname.notna()].copy()
        return df

    @property
    def _renamed(self):
        df = self._major_parties.copy()
        df = df.rename(columns={
            'state': 'name_state',
            'state_po': 'state_code',
            'county_name': 'name_county',
            'county_fips': 'fips_county',
            'candidatevotes': 'vote',
            'totalvotes': 'vote_total',
        })
        df['vote_share'] = df.vote / df.vote_total
        return df

    @property
    def _winners(self):
        df = self._renamed.copy()
        func = lambda party: df.loc[df.party == party, ['fips_county', 'vote', 'vote_share']].copy()
        df = func('DEMOCRAT').merge(func('REPUBLICAN'), on='fips_county', suffixes=('_d', '_r'))
        df.loc[df.vote_d > df.vote_r, 'r_winner'] = 0
        df.loc[df.vote_d < df.vote_r, 'r_winner'] = 1
        df.r_winner = df.r_winner.fillna(-1)
        df.r_winner = df.r_winner.apply(int)
        df['r_margin'] = df.vote_share_r - df.vote_share_d
        return df

    @property
    def _election_results(self):
        df = self._renamed.merge(self._winners, on='fips_county')
        return df


class CountyGDPParser:
    _county_gdp_filepath = _DATA_DIR + 'sources/CAGDP1__ALL_AREAS_2001_2019.csv'
    _target_year = 2019  # 2020 data not available yet via BEA
    _target_year_str = str(_target_year)

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
        dtypes.update((str(year), str) for year in range(2001, self._target_year + 1))

        df = pd.read_csv(self._county_gdp_filepath, usecols=dtypes.keys(), dtype=dtypes, encoding='windows-1252')
        df = _normalize_column_names(df)
        df.geofips = df.geofips.apply(lambda x: x.replace('"', '').strip())
        df = df[df.description == 'Current-dollar GDP (thousands of current dollars)'].copy()
        return df

    @property
    def _national_gdp(self):
        df = self._county_gdp_base.copy()
        return int(
            df.loc[df.geoname == 'United States', self._target_year_str].reset_index().loc[0, self._target_year_str])

    @property
    def _county_gdp(self):
        df = self._county_gdp_base.copy()
        years = [str(yr) for yr in range(2001, self._target_year + 1)]
        for year in years:
            df[year] = df[year].fillna(0).apply(lambda x: x.replace('(NA)', '0')).apply(int)
            df = df[df[year] > 0].copy()

        df = df.loc[:, ['geofips', 'geoname', *years]].rename(columns={'geofips': 'fips_county'})
        return df

    @property
    def _county_gdp_target_year(self):
        county_gdp_base = self._county_gdp_base.copy()
        county_gdp = county_gdp_base.loc[:, [*list(county_gdp_base.columns)[:8], self._target_year_str]]
        county_gdp[self._target_year_str] = county_gdp[self._target_year_str].fillna(0).apply(lambda x: x.replace(
            '(NA)', '0')).apply(int)
        county_gdp = county_gdp.loc[:, ['geofips', self._target_year_str]].rename(columns={
            'geofips': 'fips_county', self._target_year_str: 'gdp'})
        return county_gdp


class Summarizer(ElectionResultsParser, CountyGDPParser):
    _pop_est_filepath = _DATA_DIR + 'sources/co-est2019-alldata.csv'
    _years_elapsed = (2, 4, 8, 10)

    @property
    def _population(self):
        dtypes = {'STATE': str, 'COUNTY': str, 'POPESTIMATE2019': int}  # use 2019 population to go with 2019 GDP
        df = pd.read_csv(self._pop_est_filepath, usecols=dtypes.keys(), dtype=dtypes, encoding='windows-1252')
        df = _normalize_column_names(df)
        df.county = df.state + df.county
        df = df.drop(columns='state').rename(columns={'county': 'fips_county', 'popestimate2019': 'pop_est'})
        return df

    @property
    def _county_gdp_with_population(self):
        df = self._county_gdp_target_year.merge(self._population, on='fips_county')
        df['gdp_per_capita'] = df.gdp / df.pop_est
        return df

    @property
    def election_results_with_gdp(self):
        df = self._election_results.merge(self._county_gdp_with_population, on='fips_county')
        df['gdp_weighted_by_vote_share'] = df.gdp * df.vote_share
        df['name_county_with_state'] = [f'{c} ({s})' for c, s in zip(df.name_county, df.state_code)]
        return df

    @property
    def _county_gdp_growth(self):
        growth = self._county_gdp.copy()
        for elapsed in self._years_elapsed:
            start_yr = str(self._target_year - elapsed)
            growth[f'{elapsed}y_growth'] = growth[self._target_year_str] - growth[start_yr]
            growth[f'{elapsed}y_growth_pct'] = growth[f'{elapsed}y_growth'] / growth[start_yr]
        growth = growth.drop(columns=[f'{yr}' for yr in range(2001, self._target_year + 1)] + ['geoname'])
        return growth

    @property
    def _county_gdp_growth_with_vote_share(self):
        growth = self._county_gdp_growth.merge(self._election_results, on='fips_county')
        return growth


class OutputGenerator(Summarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results = None
        self._output = []

    def run(self):
        self._results = self.election_results_with_gdp.copy()
        self._calc_counties_won()
        self._calc_gdp_plain()
        # self._calc_gdp_weighted()
        self._calc_gdp_per_capita()
        self._calc_gdp_growth()

    def _calc_counties_won(self):
        func = lambda x: len(self._filter_by_r_winner(x))
        self._output.append({'description': 'Counties won by each major party', 'd': func(0), 'r': func(1)})

    def _calc_gdp_plain(self):
        func = lambda x: round(self._filter_by_r_winner(x)['gdp'].sum() / self._national_gdp, 3)
        self._output.append(self._create_output(
            func(0), func(1), description='Proportion of GDP accounted for by counties won by each major party'))

    def _calc_gdp_weighted(self):
        func = lambda name: round(
            self._results[self._results.candidate_shortname == name].gdp_weighted_by_vote_share.sum() /
            self._national_gdp, 3
        )
        description = (
            'Proportion of GDP accounted for by counties won by each major party - weighted by vote share instead of'
            ' using win/loss'
        )
        self._output.append(self._create_output(func('Biden'), func('Trump'), description=description))

    def _calc_gdp_per_capita(self):
        select_field = lambda x: self._filter_by_r_winner(x)['gdp_per_capita']
        func_median = lambda x: round(np.median(select_field(x)), 2)
        func_mean = lambda x: round(np.mean(select_field(x)), 2)
        func_standard_deviation = lambda x: round(np.std(select_field(x), ddof=0), 2)
        self._output.append({
            'description': 'GDP per capita across counties won by each major party',
            'd_median': func_median(0), 'r_median': func_median(1),
            'd_mean': func_mean(0), 'r_mean': func_mean(1),
            'd_standard_deviation': func_standard_deviation(0), 'r_standard_deviation': func_standard_deviation(1),
        })

    def _calc_gdp_growth(self):
        vs = self._county_gdp_growth_with_vote_share.copy()
        biden = vs[vs.candidate_shortname == 'Biden'].copy()
        output = {'description': (
            'Simple correlation at county-level of GDP growth over specified time frame to D vote share'
        )}
        for elapsed in self._years_elapsed:
            output[f'{elapsed}y_growth_pct'] = round(biden[f'{elapsed}y_growth_pct'].corr(biden.vote_share), 3)

        self._output.append(output)

    def _filter_by_r_winner(self, x):
        df = self._results[['fips_county', 'r_winner', 'gdp', 'gdp_per_capita']].drop_duplicates()
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

    @property
    def output_str(self):
        return '\n\n'.join('  \n'.join('{}: {}'.format(*i) for i in item.items()) for item in self._output)


def _normalize_column_names(df):
    return df.rename(columns=dict((col, col.strip().lower()) for col in df.columns))


def main():
    output = OutputGenerator()
    output.run()
    output.election_results_with_gdp.to_csv(_DATA_DIR + 'vote_summary.csv.with_gdp.csv', index=False)
    open(_DATA_DIR + 'output.txt', 'w').write(output.output_str)


if __name__ == '__main__':
    main()
