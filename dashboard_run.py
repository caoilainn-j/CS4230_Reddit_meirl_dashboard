# necessary imports 
import re
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Trend of Comment Behavior of Subreddit r/meirl",
    layout="wide"
)

@st.cache_data

def load_data():
    # read in initial dataset (very large, cannot push files directly to GitHub so have to insert/remove before using/pushing respectively)
    posts = pd.read_csv("meirl_datasets/the-reddit-irl-dataset-posts.csv")
    comments = pd.read_csv("meirl_datasets/the-reddit-irl-dataset-comments.csv")

    # data cleaning
    posts = posts.dropna(subset=['url'])
    comments = comments.dropna(subset=['sentiment'])

    # create dictionary containing all post IDs, with the key being the number of comments
    # Function to extract post_id from permalink
    def extract_post_id(permalink):
        if isinstance(permalink, str):
            match = re.search(r'/comments/([^/]+)/', permalink)
            if match:
                return match.group(1)
        return None

    # Extract post_id and create a new column in comments DataFrame
    comments['post_id'] = comments['permalink'].apply(extract_post_id)

    # Filter out comments where post_id could not be extracted (e.g., malformed permalinks)
    comments_with_valid_post_id = comments.dropna(subset=['post_id'])

    # Group by post_id and count comments efficiently
    comment_counts_per_post = comments_with_valid_post_id.groupby('post_id')['id'].count()

    # Initialize post_dict with all unique post IDs from the posts DataFrame, setting count to 0 initially
    post_dict = {pid: 0 for pid in posts['id'].unique()}

    # Update counts for posts that have comments
    # Only update if post_id exists in our original posts list from 'posts' DataFrame
    for post_id, count in comment_counts_per_post.items():
        if post_id in post_dict:
            post_dict[post_id] = count

    # convert earlier dict into a dataframe
    df = pd.DataFrame(list(post_dict.items()), columns=['post_id', 'num_comments'])

    # exclude posts with no comments
    df = df[df['num_comments'] > 0]

    # create appropriate intertvals
    bins = [0, 1, 5, 10, 50, 100, 500, df['num_comments'].max()]
    labels = ['1', '2-5', '6-10', '11-50', '51-100', '101-500', '500+']

    # put into bins
    df['comment_range'] = pd.cut(df['num_comments'], bins=bins, labels=labels)

    # calculate percentage of posts in each bin
    distribution = df['comment_range'].value_counts(normalize=True) * 100

    # sort for nicer output
    distribution = distribution.sort_index()

    # make a sample dataframe of 10,000 posts to use for analysis by using the ratio from above
    # total sample size
    sample_size = 10000
    # sample post dictionary
    sp_dicts = []

    # loop to go through each interval
    for bin_label, proportion in distribution.items():

        #number to sample from this bin
        n = int((proportion / 100) * sample_size)

        #get all the posts within this interval in this bin
        bin_df = df[df['comment_range'] == bin_label]

        #sample from the bins into a new dict
        sampled_bin = bin_df.sample(n=min(n, len(bin_df)), random_state=42)
        sp_dicts.append(sampled_bin)

    # combine all sampled bins
    sp_dict = pd.concat(sp_dicts)

    # shuffle to see variation in number of comments in output
    sp_dict = sp_dict.sample(frac=1, random_state=42).reset_index(drop=True)

    # separate the comments that apply to these posts to use for analysis
    # sampled post id
    sp_ids = sp_dict['post_id']

    # filter
    sampled_comments = comments[comments['post_id'].isin(sp_ids)]

    # setting up the posts dataframe with the other attributes
    # get the sample posts from posts
    sampled_posts_only = posts[posts['id'].isin(sp_ids)]

    # can change based on what we need
    post_columns = [
        'id',
        'created_utc',
        'permalink',
        'score',
    ]

    # copy to a new dataframe with the columns we need for analysis
    sampled_posts_clean = sampled_posts_only[post_columns].copy()

    # setting up the posts dataframe with the other attributes
    # attributes we take from original dataframe
    comment_columns = [
        'id',
        'created_utc',
        'permalink',
        'body',
        'sentiment',
        'post_id'
    ]

    # copy into new comments dataframe for analysis
    sampled_comments_clean = sampled_comments[comment_columns].copy()
    
    return sampled_posts_clean, sampled_comments_clean, distribution, sp_dict

@st.cache_data
def prepare_graph_data():
    sampled_posts_clean, sampled_comments_clean, distribution, sp_dict = load_data()

    network1 = nx.Graph()

    # add post nodes
    for _, row in sampled_posts_clean.iterrows():
        network1.add_node(
            row["id"],
            type="post",
            created_utc=row["created_utc"],
            permalink=row["permalink"],
            score=row["score"]
        )

    valid_post_ids = set(sampled_posts_clean["id"])

    # add comment nodes and connect to posts
    for _, row in sampled_comments_clean.iterrows():
        comment_id = row["id"]
        post_id = row["post_id"]
        sentiment = row["sentiment"]

        if post_id in valid_post_ids:
            if sentiment > 0.2:
                color = "green"
            elif sentiment < -0.2:
                color = "red"
            else:
                color = "blue"

            network1.add_node(
                comment_id,
                type="comment",
                created_utc=row["created_utc"],
                permalink=row["permalink"],
                body=row["body"],
                sentiment=sentiment,
                post_id=post_id,
                color=color
            )
            network1.add_edge(post_id, comment_id)

    return sampled_posts_clean, sampled_comments_clean, distribution, sp_dict, network1   

st.title("Trend of Comment Behavior of Subreddit r/meirl")
st.write(
    """Exploration of comment behavior of posts from subreddits r/meirl and r/me_irl,
        which are both considered "relatable humor" subreddits, looking at posts and the types 
        of comments attached to it. Here, we will see what kinds of comments appear on posts, 
        and what kinds of comments appear over time.
        In the tabs below, you can investigate the post and comment attributes, network structure,
        design and metrics, comment behavior over time for top posts in the subreddits, and conclusions
        from our investigations!
    """
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Graph Structure",
    "Sentiment Over Time",
    "Interpretation & Conclusions"
])

# page 1
with tab1:
    st.header("Overview")
    st.markdown("""
    The main idea of this page is to give you an introduction to the dataset! Our datasets come from researchers on Kaggle,
    in which they scrolled relateable humor subreddits on Reddit.com, namely r/meirl and r/me_irl, and collected posts and their
    comments in order to ask interesting questions about their behavior. While relatively niche, these subreddits in particular are
    interesting to look at because it exists within the "humor" scope, and it can pose some interesting questions
    when a lot of negativity is seen, and what that can tell us about commenting behaior on Reddit for these communities.
    You will see how many post nodes and comment nodes (that correspond to a post node) will exist in our network, where we dive deeper
    into what a post's comment section can look like in regard to "sentiment."
    
    *Sentiment* is what the researchers who collected this datset as the overall tone of a comment, with
    the value of a comment's sentiment existing between 1 (completely positive comment) and -1 (completely negative comment.)
    While sentiment isn't used usefully in this tab, it becomes more important to look at in the later tabs, as we are able to determine what a 
    post's comment section overall tone is and how it can impact what kinds of comments occur over time.          
    """)
    # load in data here
    post_nodes, comment_nodes, dist, sp_dict = load_data()
    
    st.markdown("""
    ## Research Questions
       Below are the questions that are guiding our research, and what helps us structure our network configurations, graphs, and analysis:
       #### 1. Do comments from the same post share the same sentiment?
       #### 2. Regarding sentiment, how does a post's comment section change over time?
       #### 3. Do posts that already have a lot of comments tend to attract more over time?        
    """)

    st.markdown("""
    ## Dataset Details
    """)
    c1, c2 = st.columns(2)
    c1.metric("Post Nodes", len(post_nodes))
    c2.metric("Comment Nodes", len(comment_nodes))
    st.markdown("""
    ### Comment Distribution Among Posts
    The comment distribution shows the proportion of posts (shown on the left have side) that have
    a given number of comments (shown on the right side.) This can help give an idea of what the post nodes
    in the network will look like, with more posts having fewer comments and fewer posts having a lot of comments.
            
    This also gives a sneak-peak into what the degree distribution will look like, detailed more in the next tab...
    """)
    dist = dist.to_frame()
    # distribution dropdown
    with st.expander("Comment Distribution Dropdown"):
        st.dataframe(
            dist[[
                "proportion" 
            ]],
        use_container_width=True
        )

    
    st.markdown("""
    ### Post Nodes
    The post nodes in our network are representative of the posts found in the r/meirl and r/me_irl
    subreddits. While cut down after cleaning, we have just under 10,000 posts thta all have comments attached to them, as
    posts that have no comments within this dataset are not very useful to the questions we are trying to answer.
                
    Each post node has four attributes of interest:
    1. 'id': the unique indentification value of the post
                
    2. 'created_utc': the timestamp that indicates when the post was posted in the subreddit it belongs to
    
    3. 'permalink': the link to the post on Reddit.com
                
    4. 'score': the cumulative number of upvotes/downvotes on a post
                
    Following below is a dropdown of all the posts intended for our network, along with their attributes:
    """)
    # post node dropdown
    with st.expander("Post Nodes Dropdown"):
        st.dataframe(
            post_nodes[[
                "id", "created_utc", "permalink", "score"
            ]],
        use_container_width=True
        )

    st.markdown("""
    ### Comment Nodes
    The comment nodes in our network represent the comments on a respective post found in the subreddits. This is a significantly
    larger dataset, with over 69,000 comments, each being connected to exactly one post. The attributes of interest for the comment nodes
    are:
    1. 'id': the unique indentification value of the comment
    
    2. 'created_utc': the timestamp that indicates when the comment was posted on the post it belongs to
                
    3. 'permalink': the link to the comment on Reddit.com
                
    4. 'body': the actual contents of the comment itself / what the comment says
                
    5. 'sentiment': the sentiment score (described above) of comment, indicating where on a scale (-1 to 1) the tone of a comment lies
                
    6. 'post_id': the identification value of the post the comment is posted on

    Following below is a dropdown of all the comments intended for our network, along with their attributes:          
    """)
    # comment node dropdown
    with st.expander("Comment Nodes Dropdown"):
        st.dataframe(
            comment_nodes[[
                "id", "created_utc", "permalink", "body", "sentiment", "post_id"
            ]],
        use_container_width=True
        )

# page 2
with tab2:
    st.header("Graph Structure & Degree Distribution")

    sampled_posts_clean, sampled_comments_clean, distribution, sp_dict, network1 = prepare_graph_data()

    st.markdown("""
    This tab shows two network-related views:

    - a bipartite subgraph
    - a degree distribution graph
    """)

    num_posts_to_show = st.slider(
        "Number of post nodes to show in the bipartite subgraph",
        min_value=1,
        max_value=min(25, len(sampled_posts_clean)),
        value=5
    )

    st.subheader("Bipartite Subgraph: Posts and Comments")

    # choose some post nodes
    post_nodes_subset = [
        n for n, d in network1.nodes(data=True) if d["type"] == "post"
    ][:num_posts_to_show]

    # collect selected posts and their connected comments
    nodes_to_draw = set(post_nodes_subset)
    for p in post_nodes_subset:
        nodes_to_draw.update(network1.neighbors(p))

    # create subgraph
    small_graph = network1.subgraph(nodes_to_draw)

    # separate node sets
    post_nodes_small = [n for n, d in small_graph.nodes(data=True) if d["type"] == "post"]
    comment_nodes_small = [n for n, d in small_graph.nodes(data=True) if d["type"] == "comment"]

    # bipartite layout
    bipartite_pos = nx.bipartite_layout(small_graph, post_nodes_small)

    # colors and sizes
    node_colors = []
    node_sizes = []

    for n in small_graph.nodes():
        if small_graph.nodes[n]["type"] == "post":
            node_colors.append("black")
            node_sizes.append(250)
        else:
            node_colors.append(small_graph.nodes[n].get("color", "blue"))
            node_sizes.append(80)

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    nx.draw(
        small_graph,
        bipartite_pos,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=False,
        ax=ax1
    )
    ax1.set_title("Bipartite Subgraph: Posts and Comments")

    st.pyplot(fig1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nodes", small_graph.number_of_nodes())
    c2.metric("Subgraph Edges", small_graph.number_of_edges())
    c3.metric("Post Nodes", len(post_nodes_small))
    c4.metric("Comment Nodes", len(comment_nodes_small))

    st.markdown("""
    Black nodes represent posts. Colored nodes represent comments attached to one post.
    Comment color reflects sentiment grouping:
    - green = more positive
    - red = more negative
    - blue = more neutral
    """)

    st.subheader("Degree Distribution of Post Nodes")

    degrees = sp_dict["num_comments"]
    max_degree = int(degrees.max())
    bins = range(0, max_degree + 10, 10)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(degrees, bins=bins)
    ax2.set_xlabel("Number of Comments")
    ax2.set_ylabel("Number of Posts")
    ax2.set_title("Degree Distribution of Post Nodes")

    st.pyplot(fig2)

    st.markdown("""
    Degree distribution explanation
    """)
# page 3
with tab3:
    st.header("Sentiment Here")

# page 4
with tab4:
    st.header("Conclusion Here")