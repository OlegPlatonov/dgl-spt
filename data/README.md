# Spatiotemporal datasets

This directory can be used to keep datasets for this project. If you provide a dataset name (rather than a file path)
to the `dataset` command line ergument of the `train.py` script, it will look for the dataset in this directory.

Each dataset is stored in a `.npz` file (NumPy's compressed zipped archive 
containing objects &mdash; typically, numpy arrays &mdash; and their names).

To use a dataset with our code, the following objects should be present in the `.npz` file under corresponding names:

- **`targets`**: an array of shape `[num_timestamps, num_nodes]` containing the time series values to be predicted
for each node (the dependent/endogeneous variables). Currently, only one time series per node is supported
(i.e., a time series in each node is univariate). This array may contain `NaN` values.

- **`train_timestamps`**, **`val_timestamps`**, **`test_timestamps`**: data splits &mdash; three 1-dimensional arrays
containing indices of timestamps corresponding to the train, val, and test splits respectively.
It is expected that the indices in each array are ordered and that
`len(train_timestamps) + len(val_timestamps) + len(test_timestamps) == num_timestamps`.
Even if some timestamps contain only `NaN` target values, do not drop them &mdash; this will be handled by
the data preprocessing code.

- **`spatial_node_features`**: an array of shape `[1, num_nodes, spatial_features_dim]`
containing spatial features &mdash; the features that do not change over time but are different for different nodes,
see the **Spatial, temporal, and spatiotemporal node features** section below for details.
Note that `spatial_features_dim` can be zero (if there are no spatial features in the dataset).

- **`spatial_node_feature_names`**: a 1-dimensional array of length `spatial_features_dim` containing
a name for each spatial feature, see the **Node feature names** section below for details.

- **`temporal_node_features`**: an array of shape `[num_timestamps, 1, temporal_features_dim]`
containing temporal node features &mdash; the features that change over time but are shared between all nodes,
see the **Spatial, temporal, and spatiotemporal node features** section below for details.
Note that `temporal_features_dim` can be zero (if there are no temporal features in the dataset).

- **`temporal_node_feature_names`**: a 1-dimensional array of length `temporal_features_dim` containing
a name for each temporal feature, see the **Node feature names** section below for details.

- **`spatiotemporal_node_features`**: an array of shape `[num_timestamps, num_nodes, spatiotemporal_features_dim]`
containing spatiotemporal features &mdash; the features that change over time and are different for different nodes,
see the **Spatial, temporal, and spatiotemporal node features** section below for details.
Note that `spatiotemporal_features_dim` can be zero (if there are no spatiotemporal features in the dataset).

- **`spatiotemporal_node_feature_names`**: a 1-dimensional array of length `spatiotemporal_features_dim` containing
a name for each spatiotemporal feature, see the **Node feature names** section below for details.

- **`num_feature_names`**: a 1-dimensional array of length `num_features_dim`
containing the names of numerical features.
See the **Numerical, binary, and categorical node features** section below for more details.
Note that `num_features_dim` can be zero (if there are no numerical features in the dataset).

- **`bin_feature_names`**: a 1-dimensional array of length `bin_features_dim`
containing the names of binary features.
See the **Numerical, binary, and categorical node features** section below for more details.
Note that `bin_features_dim` can be zero (if there are no binary features in the dataset).

- **`cat_feature_names`**: a 1-dimensional array of length `cat_features_dim`
containing the names of categorical features.
See the **Numerical, binary, and categorical node features** section below for more details.
Note that `cat_features_dim` can be zero (if there are no categorical features in the dataset).

- **`edges`**: an array of shape `[num_edges, 2]` containing directed edges of the graph in the edge list format.
If your graph is undirected, duplicate each edge (i.e., put both `(u, v)` and `(v, u)` edges in the array).

Other optional objects and metadata can be added to the `.npz` dataset files as well. For example, in all our datasets,
we also provide the following objects: `num_timestamps` (of type `int`) and `num_nodes` (of type `int`)
(if not provided, these two values are inferred from the `targets` array shape),
`first_timestamp_datetime` (of type `datetime.datetime`), `last_timestamp_datetime` (of type `datetime.datetime`),
`timestamp_frequency` (of type `datetime.timedelta`) (these can be used to infer the time of any timestamp),
`deepwalk_node_embeddings` (these embeddings can be used as additional node features).





### Spatial, temporal, and spatiotemporal node features

In time series forecsting, a time series is all you need &mdash; that is, you can predict future time series values
based only on past time series values. But sometimes additional data is available that can also be used as
input features for the model (predictors/exogeneous variables). In this framework, we call such data **node features**,
since we represent it as features of graph nodes. In general, such features can be stored as an array of shape
`[num_timestamps, num_nodes, features_dim]`. However, this might be highly inefficient, as there are often
features that are static (do not change over time), or features that are dynamic (do change over time) but are
the same for all nodes. Thus, for efficiency, we divide all node features into three types:
**spatial features**, **temporal features**, and **spatiotemporal features**. Different types of storage are used
for these three types of features.

**Spatiotemporal features** is the most general feature type. It is used for features that change over time and
are different for different nodes at a fixed timestamp. There is no trivial way to store them more efficiently,
so we store them as an array of shape `[num_timestamps, num_nodes, spatiotemporal_features_dim]`. Examples of such
features are different weather characteristics (temperature, precipitation, etc.) that can be used in weather modeling
or traffic modeling. These  characteristics change over time and are different in different regions of space (which can
be modeled as graph nodes).

**Spatial features** are features that do not change over time (are static), but are different for different nodes.
We store them as an array of shape `[1, num_nodes, spatial_features_dim]`, thus compressing the time dimension
of the array. Examples of such features are coordinates of traffic sensors or starts and ends of roads, as well as
static features of roads such as speed limit, which can be used in traffic modeling. These features are different
for different nodes, but do not change over time.

**Temporal features** are features that are the same for all nodes at a fixed timestamp but change over time
(are dynamic). We store them as an array of shape `[num_tiemstamps, 1, temporal_features_dim]`, thus compressing
the spatial dimension of the array. Examples of such features are hour/day/month features that can be used in
weather modeling, traffic modeling, or demand modeling, as well as holiday indicators that can be used in
traffic modeling or demand modeling. These features change over time, but have the same value for all nodes at
a fixed timestamp.





### Numerical, binary, and categorical node features

In this project, we like dividing node features into three types. Besides dividing node features into three types
based on whether thay change over temporal and/or spatial dimension (as discussed in the section above), we also divide
node features into three types based on what kinds of data they represent: numerical, binary, or categorical.
This feature classification is used to apply different kinds of preprocessing to these features.

**Numerical features** are features that take values from a continuous subset of real numbers. Examples of such features
are coordinates of traffic sensors or starts and ends of roads, or web traffic, or time spent by a user using
a particular application. When using such features as input to neural network models, it is typically helpful
to preprocess them by applying some standardization such as standard scaling, min-max scaling, or quantile transform
to normal or uniform distribution. Additionally, special [learnable embeddings](https://arxiv.org/abs/2203.05556)
can be applied to such features. This teqniques can significanly boost the performance of neural network models
and thus should not be neglected. Numerical features can contain `NaN` values &mdash; our code will impute them.

**Categorical features** are features that take values from a discrete set. Examples of such features are
days of the week, or months of the year, or counties in which roads are located, or genres of video games.
We store such features by encoding their values as integers, and then our data preprocessing code applies one-hot
encoding to them. If such features have `NaN` values, these values should simply be encoded as one more category.

**Binary features** are features that take the value of either 0 or 1. They can be viewed as a special case of
categorical features, but we classify them as a separate type since they do not require any preprocessing at all.
Examples of such features are whether it is raining, whether a road is restricted for trucks, whether a road has
a bus-only lane. If a feature that is originally binary has `NaN` values, it should be converted to a ternary feature
and treated as a categorical feature instead.





### Node feature names

To be able to easily distinguish between different features and work with different feature types, in this project,
we expect that each feature has a name. This allows us to do things like store a list of names of all numerical
features and then check if a particular feature is in this list. One more important thing is that feature names
allow us to understand what meaning a feature has, and thus guesstimate how important it is for the task and what
preprocessing it is better to apply to it (e.g., standard scaling or min-max scaling). Yes, such things can be
relatively boring, but they can often bring much larger improvements than trying a fancy new model. If you do not know
where a particular feature comes from and what meaning it has, or do not want to disclose it, or are just lazy,
simply name it `feature_1` or something like that.
