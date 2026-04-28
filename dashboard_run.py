# necessary imports 
import re
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Trend of Comment Behavior of Subreddit r/meirl",
    layout="wide"
)

@st.cache_data

def load_data():
    # read in initial dataset (very large, cannot push files directly to GitHub so have to unzip before using and zip before pushing)
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
    
    return sampled_posts_clean, sampled_comments_clean, distribution

    

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

with tab1:
    st.header("Overview Here")
    # load in data here
    post_nodes, comment_nodes, dist = load_data()

    c1, c2 = st.columns(2)
    c1.metric("Post Nodes", len(post_nodes))
    c2.metric("Comment Nodes", len(comment_nodes))
    st.markdown("""
    ### Comment Distribution Among Posts
    Explanation here
    """)
    dist = dist.to_frame()

    with st.expander("Post Nodes"):
        st.dataframe(
            dist[[
                "proportion" 
            ]],
        use_container_width=True
        )

    st.markdown("""
    ### Post Nodes
    Description of post nodes
    """)
    with st.expander("Post Nodes"):
        st.dataframe(
            post_nodes[[
                "id", "created_utc", "permalink", "score"
                
            ]],
        use_container_width=True
        )

    st.markdown("""
    ### Comment Nodes
    Description of post nodes
    """)
    with st.expander("Comment Nodes"):
        st.dataframe(
            comment_nodes[[
                "id", "created_utc", "permalink", "body", "sentiment", "post_id"
            ]],
        use_container_width=True
        )

with tab2:
    st.header("Graph Structure, Degree Distribution Here")

with tab3:
    st.header("Sentiment Here")

with tab4:
    st.header("Conclusion Here")