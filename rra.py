import pandas as pd
import numpy as np
import math


class RobustRankAggreg:
    @staticmethod
    def rank_matrix(glist, N=None, full=False):
        u = []
        seen = set()
        for sublist in glist:
            for item in sublist:
                if item not in seen:
                    u.append(item)
                    seen.add(item)

        if N is None:
            N_val = len(u)
        else:
            N_val = N

        num_lists = len(glist)

        if not full:
            rmat = pd.DataFrame(1.0, index=u, columns=range(num_lists))
            Ns = [N_val] * num_lists if isinstance(N_val, (int, float)) else N_val
        else:
            rmat = pd.DataFrame(np.nan, index=u, columns=range(num_lists))
            Ns = [len(l) for l in glist]

        for i, sublist in enumerate(glist):
            for rank_0, element in enumerate(sublist):
                rmat.loc[element, i] = (rank_0 + 1) / Ns[i]

        return rmat

    @classmethod
    def log_score(cls, scores):
        return -np.log10(scores)

    @staticmethod
    def _format_and_sort_result(scores):
        native_names = [x.item() if hasattr(x, 'item') else x for x in scores.index]
        res = pd.DataFrame({
            'Name': native_names,
            'Score': scores.values
        })

        res = res.sort_values(by='Score', ascending=False, kind='stable').reset_index(drop=True)
        return res

    @classmethod
    def aggregate_min(cls, rmat):
        scores = rmat.min(axis=1, skipna=True)
        return cls._format_and_sort_result(cls.log_score(scores))

    @classmethod
    def aggregate_mean(cls, rmat):
        a = rmat.mean(axis=1, skipna=True)
        n = rmat.notna().sum(axis=1)

        def compute_pnorm(row):
            a_val, n_val = row['a'], row['n']
            if n_val == 0 or pd.isna(a_val):
                return np.nan

            mu = 0.5
            sigma = math.sqrt(1.0 / (12.0 * n_val))
            z = (a_val - mu) / (sigma * math.sqrt(2.0))
            return 0.5 * (1.0 + math.erf(z))

        df_temp = pd.DataFrame({'a': a, 'n': n})
        scores = df_temp.apply(compute_pnorm, axis=1)

        return cls._format_and_sort_result(cls.log_score(scores))

    @classmethod
    def aggregate_geom_mean(cls, rmat):
        log_mat = np.log(rmat)
        mean_log = log_mat.mean(axis=1, skipna=True)
        scores = np.exp(mean_log)
        return cls._format_and_sort_result(cls.log_score(scores))

    @classmethod
    def aggregate_weighted_geom_mean(cls, rmat, weights):
        weights = np.array(weights)
        weights = weights / weights.sum()
        log_mat = np.log(rmat)

        weighted_sum_log = log_mat.dot(weights)
        scores = np.exp(weighted_sum_log)
        return cls._format_and_sort_result(cls.log_score(scores))

    @classmethod
    def aggregate_combmnz(cls, glist, scores_list=None, N=None, norm="z-score"):
        u = []
        seen = set()
        for sublist in glist:
            for item in sublist:
                if item not in seen:
                    u.append(item)
                    seen.add(item)

        num_lists = len(glist)

        if N is None:
            N_val = len(u)
        else:
            N_val = N
        Ns = [N_val] * num_lists if isinstance(N_val, (int, float)) else N_val

        score_mat = pd.DataFrame(np.nan, index=u, columns=range(num_lists))

        for i, sublist in enumerate(glist):
            for rank_0, element in enumerate(sublist):
                if scores_list is not None:
                    raw_score = float(scores_list[i][rank_0])
                else:
                    raw_score = 1.0 - (rank_0 / Ns[i])

                score_mat.loc[element, i] = raw_score

        if norm.lower() == "z-score":
            for col in score_mat.columns:
                col_data = score_mat[col]
                if col_data.notna().sum() > 1:
                    mu = col_data.mean()
                    sigma = col_data.std(ddof=0)

                    if sigma > 0:
                        z = (col_data - mu) / sigma
                        score_mat[col] = z
                    else:
                        score_mat[col] = 0.0
                else:
                    score_mat[col] = 0.0

        count_non_zero = score_mat.notna().sum(axis=1)
        sum_scores = score_mat.sum(axis=1, skipna=True)
        comb_scores = count_non_zero * sum_scores

        return cls._format_and_sort_result(comb_scores)

    @classmethod
    def aggregate_copeland(cls, glist):
        seen = set()
        for sublist in glist:
            for item in sublist:
                seen.add(item)
        u = sorted(list(seen))

        n_elements = len(u)
        num_lists = len(glist)

        rank_dicts = []
        sum_ranks = {item: 0.0 for item in u}
        N_total = len(u)

        for sublist in glist:
            d = {item: rank for rank, item in enumerate(sublist)}
            rank_dicts.append(d)
            for item in u:
                sum_ranks[item] += d.get(item, N_total)

        scores = {item: 0.0 for item in u}

        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                item_a = u[i]
                item_b = u[j]

                wins_a = 0
                wins_b = 0

                for r_dict in rank_dicts:
                    pos_a = r_dict.get(item_a, None)
                    pos_b = r_dict.get(item_b, None)

                    if pos_a is not None and pos_b is not None:
                        if pos_a < pos_b:
                            wins_a += 1
                        elif pos_b < pos_a:
                            wins_b += 1
                    elif pos_a is not None:
                        wins_a += 1
                    elif pos_b is not None:
                        wins_b += 1

                if wins_a > wins_b:
                    scores[item_a] += 1
                    scores[item_b] -= 1
                elif wins_b > wins_a:
                    scores[item_b] += 1
                    scores[item_a] -= 1

        res_df = pd.DataFrame({
            'Name': u,
            'Score': [scores[item] for item in u],
            'AvgRank': [sum_ranks[item] / num_lists for item in u]
        })

        res_df = res_df.sort_values(
            by=['Score', 'AvgRank', 'Name'],
            ascending=[False, True, True]
        ).reset_index(drop=True)

        return res_df[['Name', 'Score']]

    @classmethod
    def aggregate_ranks(cls, glist, scores_list=None, N=None, full=False, method="min"):

        methods_map = {
            "min": cls.aggregate_min,
            "mean": cls.aggregate_mean,
            "geom.mean": cls.aggregate_geom_mean,
            "combmnz": cls.aggregate_combmnz,
            "copeland": cls.aggregate_copeland
        }

        method_key = method.lower()
        if method_key not in methods_map:
            raise NotImplementedError(f"Method '{method}' error.")

        if method_key in ['combmnz']:
            return methods_map[method_key](glist, scores_list)
        elif method_key == 'copeland':
            return methods_map[method_key](glist)
        else:
            rmat = cls.rank_matrix(glist, N=N, full=full)
            return methods_map[method_key](rmat)


if __name__ == "__main__":
    glist_test = [
        ["a", "b", "c"], ["b", "a"],
        ["d", "c", "a"]
    ]

    print("=== 1. Min ===")
    res_min = RobustRankAggreg.aggregate_ranks(glist_test, method="min", full=False)
    print(res_min)

    res_mean = RobustRankAggreg.aggregate_ranks(glist_test, method="mean", full=True)
    print(res_mean)

    res_geom = RobustRankAggreg.aggregate_ranks(glist_test, method="geom.mean", full=False)
    print(res_geom)