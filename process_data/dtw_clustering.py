from config import *
from dtaidistance import dtw
from dtaidistance import clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster


# 对5G数据做MinMaxScaler
def scaler_5g_data(df, cols_name_5g='5g_download'):
    sector_list = df.index.unique().tolist()
    for sector in sector_list:
        tmp_df = df[df.index == sector]
        scaler = MinMaxScaler(feature_range=(0, 1))
        tmp_df['download'] = scaler.fit_transform(tmp_df[cols_name_5g].values.reshape(-1, 1))
        df.loc[sector][cols_name_5g] = tmp_df[['download']]
    return df


def format_process(df):
    data = []
    sector_list = df.index.unique()
    for i, sector in enumerate(sector_list):
        data.append(df.loc[sector]['5g_download'].values)
    return data


def plt_dtw_distance_matrix(data):
    ds = dtw.distance_matrix_fast(data)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(ds, cmap='YlGnBu', center=0, fmt='.2f',
                     square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.savefig('../figures/dtw_distance.png')
    plt.show()


def hierarchy_clusting(data, package_using='SciPy'):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    show_ts_label = lambda idx: "ts-" + str(idx)
    if (package_using == 'SciPy'):
        print("using SciPy linkage clustering")
        model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
        cluster_idx = model.fit(data)
        model.plot('../figures/hierarchy.png', axes=ax, show_ts_label=show_ts_label,
                   show_tr_label=True, ts_label_margin=-10,
                   ts_left_margin=10, ts_sample_length=1)
    else:
        model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
        # Augment Hierarchical object to keep track of the full tree
        model = clustering.HierarchicalTree(model1)
        cluster_idx = model.fit(data)
        model.plot('../figures/hierarchy.png', axes=ax, show_ts_label=show_ts_label,
                   show_tr_label=True, ts_label_margin=-10,
                   ts_left_margin=10, ts_sample_length=1)
    return cluster_idx


def fancy_dendrogram(max_d, cluster_idx, p, truncate_mode='lastp', leaf_rotation=90.,
                     leaf_font_size=12., show_contracted=True, plot=True, annotate_above=1,
                     color_threshold=None):
    if max_d and not color_threshold:
        color_threshold = max_d

    ddata = dendrogram(Z=cluster_idx, p=p, truncate_mode=truncate_mode,
                       color_threshold=color_threshold, leaf_rotation=leaf_rotation,
                       leaf_font_size=leaf_font_size, show_contracted=show_contracted)

    if plot != False:
        plt.title('Hierarchical Clustering')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * np.sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
        plt.savefig('../figures/hierachical_clustering_with_size.png')
        plt.show()
    return ddata


def get_cluster_id(cluster_idx, df, sector_list, max_d=3):
    # max_d = 3
    clusters = fcluster(cluster_idx, max_d, criterion='distance')
    cluster_df = pd.DataFrame()

    cluster_df['index'] = sector_list
    cluster_df['cluster'] = clusters
    cluster_df.set_index('index', inplace=True)

    res_df = df.merge(cluster_df, left_index=True, right_index=True)

    return res_df


if __name__ == '__main__':
    path = '../5Gprocessed_data/南京SECTOR合并.csv'
    df = pd.read_csv(path, index_col='SECTOR_ID')
    new_df = df.loc[df.index.unique()[:200]]
    sector_list = new_df.index.unique().tolist()
    new_df = scaler_5g_data(new_df)
    data = format_process(new_df)
    # plt_dtw_distance_matrix(data)
    cluster_idx = hierarchy_clusting(data)
    # p 是最下面至少保留几个
    max_d = 1

    fancy_dendrogram(max_d, cluster_idx, p=20)
    res_df = get_cluster_id(cluster_idx, df, sector_list, max_d)
    res_df.reset_index(inplace=True)
    res_df['SECTOR_ID'] = res_df['index']
    res_df.drop('index', axis=1, inplace=True)
    print(res_df)
    res_df.to_csv('../5Gprocessed_data/南京SECTOR合并_clustered.csv')
