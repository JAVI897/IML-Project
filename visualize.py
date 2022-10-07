import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
import prince

def bar_plot_vote(X, Y, labels, output, rename_target = None):
    plt.style.use('seaborn-white')
    X['Target'] = Y
    if rename_target is not None:
        X['Target'] = X['Target'].replace(rename_target)
    X['Cluster'] = labels
    for column in X.columns:
        print(column)
        fig = plt.figure(figsize=(20, 10))
        sns.countplot(data = X, x = 'Cluster', hue=column)
        plt.savefig(output+'{}.jpg'.format(column), bbox_inches='tight')

def coordinate_plot(X, Y, labels, columns_to_plot, output, rename_target=None):
    plt.style.use('seaborn-white')
    X['Target'] = Y
    if rename_target is not None:
        X['Target'] = X['Target'].replace(rename_target)
    X['Cluster'] = labels
    fig = plt.figure(figsize=(20, 10))
    parallel_coordinates(X, 'Target', cols = ['Cluster'] + columns_to_plot)
    plt.savefig(output + 'parallel_coords.jpg', bbox_inches='tight')

def coordinate_plot_by_cluster(X, Y, labels, columns_to_plot, output, rename_target=None):
    plt.style.use('seaborn-white')
    X['Target'] = Y
    if rename_target is not None:
        X['Target'] = X['Target'].replace(rename_target)
    X['Cluster'] = labels
    max_value = max(max(X[columns_to_plot[i]]) for i in range(len(columns_to_plot)))
    min_value = min(min(X[columns_to_plot[i]]) for i in range(len(columns_to_plot)))
    for cluster in X.Cluster.unique():
        X_c = X.loc[X['Cluster'] == cluster]
        fig = plt.figure(figsize=(20, 10))
        parallel_coordinates(X_c, 'Target', cols = columns_to_plot)
        plt.title('Cluster: {}'.format(cluster))
        plt.ylim(max_value, min_value)
        plt.savefig(output + 'parallel_coords_cluster_{}.jpg'.format(cluster), bbox_inches='tight')

def correspondence_analysis_plots(X, Y, labels, output, hue = 'cluster', rename_target=None):
    plt.style.use('seaborn-white')
    X.columns = [i.replace('_', '-').replace('-yes', '') for i in X.columns]
    mca = prince.MCA()
    mca = mca.fit(X)
    if hue == 'cluster':
        clusters = labels.astype(str)
        N = len(set(clusters))
    else:
        clusters = list(Y.replace(rename_target).values)
        N = len(set(clusters))
    row_coords = mca.row_coordinates(X)
    col_coords = mca.column_coordinates(X)
    one_hot_enc = mca.enc.fit(X)
    feature_names = one_hot_enc.get_feature_names_out()
    col_coords['features'] = feature_names
    col_coords = col_coords.set_index('features')

    fig, ax = plt.subplots(figsize=(20, 10))
    row_coords['groups'] = clusters
    for group, group_row_coords in row_coords.groupby('groups'):
        ax.scatter(
            group_row_coords.iloc[:, 0],
            group_row_coords.iloc[:, 1],
            s=10,
            label=group,
            alpha = 0.8
        )

    x = col_coords[0]
    y = col_coords[1]

    prefixes = col_coords.index.str.split('_').map(lambda x: x[0])
    for prefix in prefixes.unique():
        mask = prefixes == prefix
        ax.scatter(x[mask], y[mask], s=10, marker='X', label=prefix)
        for i, label in enumerate(col_coords[mask].index):
            ax.annotate(label, (x[mask][i], y[mask][i]))

    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:N], l[:N])
    ax.set_title('Row and column principal coordinates colored by {}'.format(hue))
    ei = mca.explained_inertia_
    ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(0, 100 * ei[0]))
    ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(1, 100 * ei[1]))

    ax.get_figure().savefig(output + 'coordinates_{}.jpg'.format(hue))
    plt.close()



